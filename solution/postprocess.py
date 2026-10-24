"""Обучает и применяет модель с калибровкой риска ошибки отдельных символов."""

import csv
import datetime as dt
import json
import math
import random
import re
import statistics
import sys
from difflib import SequenceMatcher
from pathlib import Path

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


# Все пути строятся от каталога проекта, а не от текущего каталога запуска.
ROOT = Path(__file__).resolve().parent.parent
REFERENCE_PATH = ROOT / "reference" / "subdivisions.csv"

# Фиксируем генератор случайных чисел для воспроизводимого разбиения и обучения.
SEED = 417

# Порог похожести названия с вариантами для распознанного кода подразделения.
MIN_ORGANIZATION_NAME_SIMILARITY = 0.65

# На calibration максимизируем покрытие при нуле ошибочно принятых документов.
TARGET_DOCUMENT_PRECISION = 1.0

# Кандидаты коэффициента влияния признаков char_conf для калибровки.
CHAR_CONFIDENCE_SCALES = [index / 20 for index in range(21)]

# Имена полей и порядок соответствуют формату train_labels.jsonl.
FIELDS = [
    "Last_name_ru",
    "First_name_ru",
    "Middle_name_ru",
    "Sex_ru",
    "Birth_date",
    "Birth_place_ru",
    "Issue_date",
    "Issue_organisation_code",
    "Licence_number",
    "Issue_organization_ru",
]


if nn is not None:
    class OCRClassifier(nn.Module):
        """MLP-классификатор точности каждого OCR-поля документа."""

        def __init__(self, input_size, output_size=len(FIELDS)):
            """Создаёт сеть с отдельным выходом для каждого OCR-поля."""
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Dropout(0.20),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(32, output_size),
            )

        def forward(self, values):
            """Вычисляет логит правильности каждого поля в батче."""
            return self.layers(values)


def load_jsonl(path):
    """Загружает JSONL-файл в список словарей."""

    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def normalize_text(value):
    """Нормализует текст для сравнения без учёта регистра и лишних пробелов"""
    return " ".join(str(value or "").split()).upper()


def load_subdivisions(path):
    """Загружает варианты названий подразделений по цифровому коду"""

    result = {}
    if not path or not path.exists():
        return result

    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            code = re.sub(r"\D", "", row.get("code", ""))
            name = normalize_text(row.get("name", ""))
            if code and name:
                result.setdefault(code, set()).add(name)

    return result


def mrz_lines(record):
    """Извлекает текстовые строки машиночитаемой зоны паспорта"""

    return [
        str(line.get("text", ""))
        for line in (record.get("mrz") or {}).get("lines", [])
        if isinstance(line, dict)
    ]


def mrz_date(raw):
    """Преобразует дату MRZ YYMMDD в дату с учётом допустимого диапазона"""

    if not re.fullmatch(r"\d{6}", raw or ""):
        return None

    year_part, month, day = int(raw[:2]), int(raw[2:4]), int(raw[4:6])
    candidates = []
    for century in (1900, 2000):
        try:
            value = dt.date(century + year_part, month, day)
        except ValueError:
            continue
        if dt.date(1900, 1, 1) <= value <= dt.date(2026, 10, 1):
            candidates.append(value)

    return candidates[0] if len(candidates) == 1 else (
        max(candidates) if candidates else None
    )


def date_value(value):
    """Разбирает дату в формате ДД.ММ.ГГГГ, возвращает None при ошибке"""

    try:
        return dt.datetime.strptime(value, "%d.%m.%Y").date()
    except (TypeError, ValueError):
        return None


def mrz_details(record):
    """Извлекает номер, дату рождения, пол и срок действия из второй строки MRZ"""

    lines = mrz_lines(record)
    if len(lines) < 2:
        return None

    line = re.sub(r"\s", "", lines[1]).upper()
    if len(line) < 28:
        return None

    digit_translation = str.maketrans({
        "O": "0", "Q": "0", "D": "0", "I": "1",
        "L": "1", "Z": "2", "S": "5", "B": "8",
    })
    digits = lambda value: value.translate(digit_translation)
    
    return {
        "number": digits(line[:9]),
        "birth": mrz_date(digits(line[13:19])),
        "sex": line[20:21],
        "expiry": mrz_date(digits(line[21:27])),
        "issuer_code": digits(line[28:35]) if len(line) >= 35 else None,
    }


def passport_valid(fields, details):
    """Проверяет даты паспорта и согласованность пола и даты с MRZ"""

    today = dt.date(2026, 10, 1)
    birth = date_value(fields.get("Birth_date"))
    issue = date_value(fields.get("Issue_date"))
    if not birth or not issue or issue > today or birth > issue:
        return False

    age_at_issue = issue.year - birth.year - (
        (issue.month, issue.day) < (birth.month, birth.day)
    )
    if age_at_issue not in (14, 20, 45):
        return False

    for threshold_age in (20, 45):
        try:
            birthday = birth.replace(year=birth.year + threshold_age)
        except ValueError:
            birthday = birth.replace(year=birth.year + threshold_age, day=28)
        if issue < birthday and today > birthday + dt.timedelta(days=90):
            return False

    if details:
        if details["birth"] and details["birth"] != birth:
            return False
        mrz_sex = details["sex"]
        visual_sex = normalize_text(fields.get("Sex_ru"))
        if mrz_sex in ("M", "F") and (
            (mrz_sex == "M") != visual_sex.startswith("МУЖ")
        ):
            return False

    return True


def extract_field_text(field):
    """Объединяет распознанные строки одного поля в одну строку"""

    if not isinstance(field, dict):
        return ""

    lines = [
        str(line.get("text", ""))
        for line in field.get("lines", [])
        if isinstance(line, dict)
    ]
    return " ".join(lines).strip()


def extract_mrz_features(record, fields):
    """Создаёт признаки наличия MRZ и согласованности даты рождения и пола"""

    lines = [
        str(line.get("text", ""))
        for line in (record.get("mrz") or {}).get("lines", [])
        if isinstance(line, dict)
    ]
    features = {
        "mrz_present": float(bool(lines)),
        "mrz_two_lines": float(len(lines) >= 2),
        "mrz_birth_match": 0.0,
        "mrz_birth_conflict": 0.0,
        "mrz_sex_match": 0.0,
        "mrz_sex_conflict": 0.0,
    }
    if len(lines) < 2:
        return features

    # В строке TD3 дата рождения занимает позиции 13–18, пол — позицию 20
    second_line = re.sub(r"\s", "", lines[1]).upper()
    if len(second_line) < 21:
        return features

    # В числовых позициях MRZ заменяем похожие латинские буквы на цифры,
    # чтобы сравнение не зависело от частых OCR-путаниц вроде O/0 и I/1
    digit_translation = str.maketrans(
        {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
         "Z": "2", "S": "5", "B": "8"}
    )
    raw_birth = second_line[13:19].translate(digit_translation)
    if re.fullmatch(r"\d{6}", raw_birth):
        try:
            visual_birth = dt.datetime.strptime(
                fields.get("Birth_date", ""), "%d.%m.%Y"
            ).date()
        except ValueError:
            visual_birth = None

        candidate_dates = []
        for century in (1900, 2000):
            try:
                candidate_dates.append(
                    dt.date(
                        century + int(raw_birth[:2]),
                        int(raw_birth[2:4]),
                        int(raw_birth[4:6]),
                    )
                )
            except ValueError:
                continue

        if visual_birth and candidate_dates:
            matches = [date for date in candidate_dates if date == visual_birth]
            features["mrz_birth_match"] = float(bool(matches))
            features["mrz_birth_conflict"] = float(not matches)

    mrz_sex = second_line[20:21]
    visual_sex = normalize_text(fields.get("Sex_ru"))
    if mrz_sex in ("M", "F") and visual_sex:
        agrees = (
            (mrz_sex == "M" and visual_sex.startswith("МУЖ"))
            or (mrz_sex == "F" and visual_sex.startswith("ЖЕН"))
        )
        features["mrz_sex_match"] = float(agrees)
        features["mrz_sex_conflict"] = float(not agrees)

    return features


def has_valid_format(field_name, value):
    """Проверяет базовый формат даты, пола, кода или номера документа"""

    if field_name in ("Birth_date", "Issue_date"):
        try:
            dt.datetime.strptime(value, "%d.%m.%Y")
            return True
        except ValueError:
            return False

    if field_name == "Sex_ru":
        return normalize_text(value) in ("МУЖ.", "ЖЕН.")

    if field_name == "Issue_organisation_code":
        return bool(re.fullmatch(r"\d{3}-\d{3}", value))

    if field_name == "Licence_number":
        return bool(re.fullmatch(r"\d{2}\s\d{2}\s\d{6}", value))

    return bool(value.strip())


def extract_features(record, subdivisions=None):
    """Извлекает числовые признаки OCR, метаданные снимка и распознанные поля"""

    visual_zone = record.get("viz") or {}
    fields = {
        name: extract_field_text(visual_zone.get(name))
        for name in FIELDS
    }
    features = {}

    # Метрики качества и источник снимка дают модели контекст распознавания
    quality = record.get("quality") or {}
    features["quality_doc_px"] = float(quality.get("doc_px") or 0)
    features["quality_sharpness"] = float(quality.get("sharpness") or 0)
    features["quality_glare"] = float(quality.get("glare") or 0)

    source = record.get("source", "")
    for source_name in ("web", "mobile", "mobile_v2"):
        features[f"source_{source_name}"] = float(source == source_name)

    # Общие списки нужны для агрегированных признаков по всему документу
    all_field_confidences = []
    all_character_confidences = []

    # Для каждого поля собираем полноту, confidence, статистики символов,
    # число строк, длину текста и простой признак корректного формата
    for name in FIELDS:
        field = visual_zone.get(name)
        prefix = f"field_{name}"
        features[f"{prefix}_present"] = float(isinstance(field, dict))

        if not isinstance(field, dict):
            confidence = 0.0
            lines = []
            character_confidences = []
        else:
            confidence = float(field.get("conf") or 0)
            lines = [
                line for line in field.get("lines", [])
                if isinstance(line, dict)
            ]
            character_confidences = [
                float(score)
                for line in lines
                for score in (line.get("char_conf") or [])
                if isinstance(score, (float, int))
            ]

        features[f"{prefix}_conf"] = confidence
        features[f"{prefix}_char_min"] = (
            min(character_confidences) if character_confidences else 0.0
        )
        features[f"{prefix}_char_mean"] = (
            statistics.fmean(character_confidences)
            if character_confidences else 0.0
        )
        # Нижний дециль показывает слабые оценки символов, не сводя поле
        # к единственному минимальному значению
        features[f"{prefix}_char_p10"] = (
            sorted(character_confidences)[
                max(0, int((len(character_confidences) - 1) * 0.1))
            ]
            if character_confidences else 0.0
        )
        features[f"{prefix}_line_count"] = float(len(lines))
        features[f"{prefix}_text_length"] = float(len(fields[name]))
        features[f"{prefix}_format_ok"] = float(
            has_valid_format(name, fields[name])
        )

        if isinstance(field, dict) and isinstance(field.get("conf"), (float, int)):
            all_field_confidences.append(confidence)
        all_character_confidences.extend(character_confidences)

    # Агрегаты помогают сети заметить одно особенно слабое поле документа.
    features["all_conf_min"] = (
        min(all_field_confidences)
        if len(all_field_confidences) == len(FIELDS) else 0.0
    )
    features["all_conf_mean"] = (
        statistics.fmean(all_field_confidences)
        if all_field_confidences else 0.0
    )
    features["all_char_min"] = (
        min(all_character_confidences) if all_character_confidences else 0.0
    )
    features["all_char_mean"] = (
        statistics.fmean(all_character_confidences)
        if all_character_confidences else 0.0
    )
    features.update(extract_mrz_features(record, fields))

    # Справочник даёт независимую подсказку для проверки названия органа.
    code = re.sub(r"\D", "", fields["Issue_organisation_code"])
    variants = (subdivisions or {}).get(code, set())
    normalized_name = normalize_organization_name(
        fields["Issue_organization_ru"]
    )
    normalized_variants = [
        normalize_organization_name(variant)
        for variant in variants
    ]
    features["organization_code_known"] = float(bool(variants))
    features["organization_name_similarity"] = max(
        (
            SequenceMatcher(None, normalized_name, variant).ratio()
            for variant in normalized_variants
        ),
        default=0.0,
    )
    features["organization_code_name_match"] = float(
        organization_name_matches(
            code,
            fields["Issue_organization_ru"],
            subdivisions or {},
        )
    )

    return features, fields


def field_correctness_targets(predicted_fields, label):
    """Создаёт отдельную бинарную метку правильности для каждого поля"""

    expected_fields = label["fields"]
    # Каждое поле обучается независимо: одна ошибка не портит метки остальных.
    return [
        int(
            normalize_text(predicted_fields.get(name))
            == normalize_text(expected_fields.get(name))
        )
        for name in FIELDS
    ]


def train_torch(
    feature_rows,
    labels,
    feature_names,
    char_confidence_scale=1.0,
    seed=SEED,
):
    """Обучает многовыходную MLP предсказывать точность отдельных полей"""

    if torch is None or nn is None:
        raise SystemExit(
            "Для обучения установите PyTorch: python3 -m pip install torch"
        )

    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    # Модель и тензоры явно размещаются на CPU для совместимости с заданием.
    inputs = torch.tensor(feature_rows, dtype=torch.float32, device="cpu")
    targets = torch.tensor(labels, dtype=torch.float32, device="cpu")

    # Стандартизируем признаки для ускорения обучения, параметры сохраняются
    # вместе с весами и повторно применяются при тестировании
    means = inputs.mean(dim=0)
    scales = inputs.std(dim=0, unbiased=False).clamp_min(1e-6)
    inputs = (inputs - means) / scales
    char_confidence_ids = [
        index for index, name in enumerate(feature_names)
        if "_char_" in name or name.startswith("all_char_")
    ]
    if char_confidence_ids:
        inputs[:, char_confidence_ids] *= char_confidence_scale

    sample_order = list(range(len(labels))) # Создаем список индексов выборки и перемешиваем
    random.Random(seed + 1).shuffle(sample_order) 

    # Разделяем fit данные на 2 выборки: обучающая и проверочная
    validation_size = max(1, int(len(sample_order) * 0.15)) # Вычисляем до 15% из выборки
    validation_ids = sample_order[:validation_size] # Берем эти 15% и создаем проверочную выборку
    training_ids = sample_order[validation_size:] # И создаем обучающую выборку

    model = OCRClassifier(inputs.shape[1], len(FIELDS)).cpu()

    # AdamW обучает веса, а weight_decay ограничивает переобучение на небольшом наборе
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.02,
    )
    loss_function = nn.BCEWithLogitsLoss()
    best_validation_loss = float("inf")
    best_state = None

    # Останавливаем обучение, если validation loss долго не улучшается
    patience = 30
    stale_epochs = 0
    batch_size = 128

    for epoch in range(400):
        model.train()

        # Перемешиваем обучающие примеры отдельно в каждой эпохе
        shuffled_ids = training_ids[:]
        random.Random(seed + epoch + 2).shuffle(shuffled_ids)

        # Вычисляем веса каждого батча
        for offset in range(0, len(shuffled_ids), batch_size):
            batch_ids = shuffled_ids[offset:offset + batch_size]
            optimizer.zero_grad()
            loss = loss_function(model(inputs[batch_ids]), targets[batch_ids])
            loss.backward()
            optimizer.step()

        # После обучения всех батчей эпохи, проверяем качество на проверочной выборке
        model.eval() # Переключаем сеть в режим оценки
        with torch.no_grad(): # Отключаем обновление весов
            validation_loss = float(
                loss_function(model(inputs[validation_ids]), targets[validation_ids])
            )

        # Сохраняем веса лучшей эпохи или прекращаем обучение при достижении лимита эпох
        if validation_loss < best_validation_loss - 1e-5:
            best_validation_loss = validation_loss
            best_state = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    return {
        "feature_names": feature_names,
        "char_confidence_scale": char_confidence_scale,
        "means": means.tolist(),
        "scales": scales.tolist(),
        "state_dict": {
            name: tensor.detach().cpu().tolist()
            for name, tensor in model.state_dict().items()
        },
        "architecture": [len(feature_names), 64, 32, len(FIELDS)],
        "best_internal_validation_loss": best_validation_loss,
        "epochs_trained": epoch + 1,
        "seed": seed,
    }


def predict_many(model_data, feature_rows):
    """Возвращает оценки правильности каждого поля для списка документов"""

    means = model_data["means"]
    scales = model_data["scales"]
    char_confidence_scale = float(model_data.get("char_confidence_scale", 1.0))
    char_confidence_ids = {
        index for index, name in enumerate(model_data["feature_names"])
        if "_char_" in name or name.startswith("all_char_")
    }

    if torch is None:
        # Для применения готовой модели достаточно стандартной библиотеки Python.
        state = model_data["state_dict"]
        layers = (
            (state["layers.0.weight"], state["layers.0.bias"], True),
            (state["layers.3.weight"], state["layers.3.bias"], True),
            (state["layers.6.weight"], state["layers.6.bias"], False),
        )
        predictions = []
        for row in feature_rows:
            values = []
            for index, value in enumerate(row):
                normalized = (float(value) - means[index]) / scales[index]
                if index in char_confidence_ids:
                    normalized *= char_confidence_scale
                values.append(normalized)

            for weights, biases, use_relu in layers:
                values = [
                    bias + sum(
                        weight * value
                        for weight, value in zip(weights_for_output, values)
                    )
                    for weights_for_output, bias in zip(weights, biases)
                ]
                if use_relu:
                    values = [max(0.0, value) for value in values]

            predictions.append([
                1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))
                for value in values
            ])
        return predictions

    # Восстанавливаем архитектуру и веса из JSON-модели.
    model = OCRClassifier(
        len(model_data["feature_names"]),
        len(FIELDS),
    )
    state = {
        name: torch.tensor(values, dtype=torch.float32)
        for name, values in model_data["state_dict"].items()
    }
    model.load_state_dict(state)
    model.eval()

    inputs = torch.tensor(feature_rows, dtype=torch.float32)
    means = torch.tensor(model_data["means"], dtype=torch.float32)
    scales = torch.tensor(model_data["scales"], dtype=torch.float32)
    # Используем ту же стандартизацию, что и во время обучения
    normalized_inputs = (inputs - means) / scales

    # Коэффициент применяется после стандартизации, чтобы не изменить масштаб
    # признаков при нормализации и управлять именно их влиянием на результат.
    char_confidence_ids = sorted(char_confidence_ids)
    if char_confidence_ids:
        normalized_inputs[:, char_confidence_ids] *= char_confidence_scale

    # Каждая sigmoid-оценка относится к правильности отдельного поля.
    with torch.no_grad():
        return torch.sigmoid(model(normalized_inputs)).tolist()


def select_document_threshold(scores, labels, eligible):
    """Максимизирует покрытие при целевой точности на calibration"""

    best = None
    for threshold in sorted(
        {score for score, can_accept in zip(scores, eligible) if can_accept}
    ):
        accepted_ids = [
            index
            for index, (score, can_accept) in enumerate(zip(scores, eligible))
            if can_accept and score >= threshold
        ]
        if not accepted_ids:
            continue

        precision = sum(labels[index] for index in accepted_ids) / len(accepted_ids)
        if precision < TARGET_DOCUMENT_PRECISION:
            continue

        candidate = (len(accepted_ids), -threshold, threshold)
        if best is None or candidate > best:
            best = candidate

    # Если калибровка не находит порога с целевой точностью, отклоняем всё.
    return best[2] if best else math.nextafter(1.0, math.inf)


def select_document_and_risk_thresholds(
    document_scores,
    labels,
    character_risks,
    eligible,
):
    """Подбирает пороги оценки документа и посимвольного риска"""

    candidate_risk_thresholds = sorted({
        risk
        for risk, can_accept in zip(character_risks, eligible)
        if can_accept
    })
    best = None

    for risk_threshold in candidate_risk_thresholds:
        candidates = sorted(
            (
                (document_scores[index], labels[index])
                for index, can_accept in enumerate(eligible)
                if can_accept
                and character_risks[index] <= risk_threshold
            ),
            reverse=True,
        )
        accepted_count = 0
        correct_count = 0
        offset = 0

        # Добавляем одинаковые оценки группой, чтобы не делить равные баллы.
        while offset < len(candidates):
            document_threshold = candidates[offset][0]
            while (
                offset < len(candidates)
                and candidates[offset][0] == document_threshold
            ):
                correct_count += candidates[offset][1]
                accepted_count += 1
                offset += 1

            precision = correct_count / accepted_count
            if precision < TARGET_DOCUMENT_PRECISION:
                continue

            # При равном покрытии выбираем более мягкие пороги.
            candidate = (
                accepted_count,
                -document_threshold,
                -risk_threshold,
                document_threshold,
                risk_threshold,
            )
            if best is None or candidate > best:
                best = candidate

    if best is None:
        return math.nextafter(1.0, math.inf), math.nextafter(
            max(character_risks, default=1.0), math.inf
        )
    return best[3], best[4]


def select_char_confidence_scale(
    fit_features,
    fit_labels,
    feature_names,
    feature_rows,
    samples,
    labels,
    subdivisions,
):
    """Подбирает влияние char_conf и порог оценки документа"""

    best = None
    for scale in CHAR_CONFIDENCE_SCALES:
        candidate_model = train_torch(
            fit_features,
            fit_labels,
            feature_names,
            char_confidence_scale=scale,
        )
        scores = predict_many(candidate_model, feature_rows)
        deterministic_rejections = evaluate_sample_rejections(
            samples,
            scores,
            -1.0,
            subdivisions,
        )
        eligible = [not reasons for reasons in deterministic_rejections]
        document_scores = [min(row) for row in scores]
        document_labels = [int(all(target)) for target in labels]
        threshold = select_document_threshold(
            document_scores,
            document_labels,
            eligible,
        )
        rejection_results = evaluate_sample_rejections(
            samples,
            scores,
            threshold,
            subdivisions,
        )
        metrics = summarize_decisions(rejection_results, document_labels)
        # При равном числе принятых выбираем меньший коэффициент: это
        # ослабляет char_conf, не ухудшая наблюдаемое покрытие calibration.
        candidate = (
            metrics["accepted"],
            -scale,
            candidate_model,
            threshold,
        )
        if best is None or candidate > best:
            best = candidate

    _, negative_scale, selected_model, selected_threshold = best
    return selected_model, -negative_scale, selected_threshold


def minimum_character_confidence(record):
    """Возвращает минимальную char_conf среди символов обязательных полей"""

    visual_zone = record.get("viz") or {}
    confidences = []
    for name in FIELDS:
        field = visual_zone.get(name)
        if not isinstance(field, dict):
            continue
        for line in field.get("lines", []):
            if not isinstance(line, dict):
                continue
            confidences.extend(
                float(value)
                for value in (line.get("char_conf") or [])
                if isinstance(value, (int, float))
            )
    return min(confidences) if confidences else 0.0


def field_characters_and_confidences(record, field_name):
    """Возвращает символы OCR-поля и соответствующие им оценки char_conf"""

    field = (record.get("viz") or {}).get(field_name)
    if not isinstance(field, dict):
        return "", []

    characters = []
    confidences = []
    for line in field.get("lines", []):
        if not isinstance(line, dict):
            continue
        if characters:
            characters.append(" ")
            confidences.append(1.0)

        text = str(line.get("text", "")).upper()
        scores = line.get("char_conf") or []
        for index, character in enumerate(text):
            confidence = (
                float(scores[index])
                if index < len(scores) and isinstance(scores[index], (int, float))
                else 0.0
            )
            if character.isspace():
                if characters and characters[-1] != " ":
                    characters.append(" ")
                    confidences.append(1.0)
                continue
            characters.append(character)
            confidences.append(confidence)

    while characters and characters[-1] == " ":
        characters.pop()
        confidences.pop()
    return "".join(characters), confidences


def align_character_errors(predicted_text, confidences, expected_text):
    """Сопоставляет OCR-символы с разметкой и помечает ошибки распознавания"""

    predicted = normalize_text(predicted_text)
    expected = normalize_text(expected_text)
    if len(confidences) != len(predicted):
        confidences = list(confidences[:len(predicted)])
        confidences.extend([0.0] * (len(predicted) - len(confidences)))

    rows = len(predicted) + 1
    columns = len(expected) + 1
    distances = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        distances[row][0] = row
    for column in range(columns):
        distances[0][column] = column

    for row in range(1, rows):
        for column in range(1, columns):
            substitution_cost = int(predicted[row - 1] != expected[column - 1])
            distances[row][column] = min(
                distances[row - 1][column] + 1,
                distances[row][column - 1] + 1,
                distances[row - 1][column - 1] + substitution_cost,
            )

    observations = []
    row = len(predicted)
    column = len(expected)
    while row or column:
        if row and column:
            substitution_cost = int(predicted[row - 1] != expected[column - 1])
            if distances[row][column] == (
                distances[row - 1][column - 1] + substitution_cost
            ):
                observations.append((
                    confidences[row - 1],
                    substitution_cost,
                ))
                row -= 1
                column -= 1
                continue
        if row and distances[row][column] == distances[row - 1][column] + 1:
            # Лишний символ OCR считается ошибочным символом.
            observations.append((confidences[row - 1], 1))
            row -= 1
            continue
        # Пропущенный OCR-символ представлен событием с нулевой уверенностью.
        observations.append((0.0, 1))
        column -= 1

    observations.reverse()
    return observations


def fit_character_error_calibrator(observations, bin_count=20):
    """Строит монотонную оценку ошибки символа по confidence-бинам"""

    counts = [0] * bin_count
    errors = [0] * bin_count
    for confidence, is_error in observations:
        confidence = max(0.0, min(1.0, float(confidence)))
        bin_index = min(bin_count - 1, int(confidence * bin_count))
        counts[bin_index] += 1
        errors[bin_index] += int(bool(is_error))

    # Сглаживание не позволяет маленьким бинам получить вероятность ровно 0/1.
    blocks = []
    for index, (count, error_count) in enumerate(zip(counts, errors)):
        weight = count + 2
        rate = (error_count + 1) / weight
        blocks.append({
            "start": index,
            "end": index,
            "weight": weight,
            "rate": rate,
        })
        # Ошибок должно становиться не больше при росте уверенности.
        while len(blocks) >= 2 and blocks[-2]["rate"] < blocks[-1]["rate"]:
            right = blocks.pop()
            left = blocks.pop()
            weight = left["weight"] + right["weight"]
            blocks.append({
                "start": left["start"],
                "end": right["end"],
                "weight": weight,
                "rate": (
                    left["rate"] * left["weight"]
                    + right["rate"] * right["weight"]
                ) / weight,
            })

    error_rates = [0.0] * bin_count
    for block in blocks:
        for index in range(block["start"], block["end"] + 1):
            error_rates[index] = block["rate"]

    return {
        "bin_count": bin_count,
        "counts": counts,
        "errors": errors,
        "error_rates": error_rates,
    }


def character_error_probability(confidence, calibrator):
    """Возвращает обученную вероятность ошибки символа"""

    bin_count = int(calibrator["bin_count"])
    confidence = max(0.0, min(1.0, float(confidence)))
    bin_index = min(bin_count - 1, int(confidence * bin_count))
    return float(calibrator["error_rates"][bin_index])


def document_character_error_risk(record, calibrator):
    """Оценивает риск хотя бы одной ошибки по уверенности всех символов"""

    log_probability_no_errors = 0.0
    character_count = 0
    for name in FIELDS:
        _, confidences = field_characters_and_confidences(record, name)
        for confidence in confidences:
            error_probability = character_error_probability(
                confidence,
                calibrator,
            )
            error_probability = min(1.0 - 1e-12, max(0.0, error_probability))
            log_probability_no_errors += math.log1p(-error_probability)
            character_count += 1

    if not character_count:
        return 1.0
    return min(1.0, max(0.0, 1.0 - math.exp(log_probability_no_errors)))


def character_observations_for_record(record, expected_fields):
    """Собирает обучающие пары уверенности символа и факта его ошибки"""

    observations = []
    for name in FIELDS:
        predicted_text, confidences = field_characters_and_confidences(
            record,
            name,
        )
        observations.extend(align_character_errors(
            predicted_text,
            confidences,
            expected_fields.get(name, ""),
        ))
    return observations


def normalize_organization_name(value):
    """Нормализует сокращения и знаки препинания в названии органа"""

    value = normalize_text(value).replace("Ё", "Е")
    # В OCR встречаются обе формы: «области» и сокращение «обл.».
    value = re.sub(r"\bОБЛАСТ(?:Ь|И)\b", "ОБЛ", value)
    value = re.sub(r"\bР\s*[-.]?\s*НЕ\b", "РАЙОНЕ", value)
    value = re.sub(r"\bРАЙОН\b", "РАЙОНЕ", value)
    return " ".join(re.findall(r"[А-Я0-9]+", value))


def organization_name_matches(code, name, subdivisions):
    """Проверяет название по вариантам справочника для указанного кода"""

    normalized_name = normalize_organization_name(name)
    name_tokens = normalized_name.split()
    variants = subdivisions.get(re.sub(r"\D", "", str(code)), set())
    normalized_variants = [
        normalize_organization_name(variant)
        for variant in variants
    ]
    normalized_variants = [variant for variant in normalized_variants if variant]

    if not normalized_name or not normalized_variants:
        return False
    if normalized_name in normalized_variants:
        return True

    # Если распознан точный хвост названия, значит его начальные слова пропущены.
    for variant in normalized_variants:
        variant_tokens = variant.split()
        if any(
            variant_tokens[offset:] == name_tokens
            for offset in range(1, len(variant_tokens))
        ):
            return False

    # Разрешаем отличия в сокращениях и OCR, но требуем сходства с вариантом
    # именно этого кода, а не с произвольным названием из справочника.
    similarity = max(
        SequenceMatcher(None, normalized_name, variant).ratio()
        for variant in normalized_variants
    )
    return similarity >= MIN_ORGANIZATION_NAME_SIMILARITY


def record_organization_name_matches(record, subdivisions):
    """Сверяет OCR-код и название подразделения со справочником"""

    _, fields = extract_features(record)
    code = re.sub(r"\D", "", fields["Issue_organisation_code"])
    return organization_name_matches(
        code,
        fields["Issue_organization_ru"],
        subdivisions,
    )


def rejection_reasons(
    record,
    fields,
    field_scores,
    threshold,
    subdivisions,
    character_risk=None,
    risk_threshold=None,
):
    """Проверяет оценки отдельных полей и документные правила валидности"""

    reasons = []
    weakest_field_index = min(
        range(len(field_scores)),
        key=field_scores.__getitem__,
    )
    weakest_field_score = field_scores[weakest_field_index]
    if weakest_field_score < threshold:
        reasons.append({
            "code": "weakest_field_confidence_below_threshold",
            "message": "Оценка самого слабого поля ниже порога документа",
            "field": FIELDS[weakest_field_index],
            "value": weakest_field_score,
            "threshold": threshold,
        })
    if character_risk is not None and risk_threshold is not None and character_risk > risk_threshold:
        reasons.append({
            "code": "character_error_risk_above_threshold",
            "message": "Расчётный риск ошибки по символам выше порога",
            "value": character_risk,
            "threshold": risk_threshold,
        })
    missing_fields = [name for name in FIELDS if not fields[name]]
    if missing_fields:
        reasons.append({
            "code": "missing_fields",
            "message": "Не заполнены обязательные поля",
            "fields": missing_fields,
        })

    invalid_format_fields = [
        name for name in FIELDS
        if not has_valid_format(name, fields[name])
    ]
    if invalid_format_fields:
        reasons.append({
            "code": "invalid_field_format",
            "message": "У полей неверный формат",
            "fields": invalid_format_fields,
        })

    try:
        birth_date = dt.datetime.strptime(
            fields["Birth_date"], "%d.%m.%Y"
        ).date()
        issue_date = dt.datetime.strptime(
            fields["Issue_date"], "%d.%m.%Y"
        ).date()
        dates_are_valid = birth_date <= issue_date <= dt.date(2026, 10, 1)
    except ValueError:
        dates_are_valid = False
    if not dates_are_valid:
        reasons.append({
            "code": "invalid_date_order_or_range",
            "message": "Даты не согласуются или выходят за допустимый диапазон",
        })

    # Для справочника сравниваем числовую часть кода подразделения.
    code = re.sub(r"\D", "", fields["Issue_organisation_code"])
    if code not in subdivisions:
        reasons.append({
            "code": "unknown_subdivision_code",
            "message": "Код подразделения отсутствует в справочнике",
            "value": code,
        })
    elif not organization_name_matches(
        code,
        fields["Issue_organization_ru"],
        subdivisions,
    ):
        reasons.append({
            "code": "organization_name_does_not_match_code",
            "message": "Название подразделения не совпадает с кодом",
            "value": fields["Issue_organization_ru"],
            "subdivision_code": code,
        })

    if not passport_valid(fields, mrz_details(record)):
        reasons.append({
            "code": "passport_validity_check_failed",
            "message": "Паспорт не прошёл проверку валидности",
        })
    return reasons


def evaluate_sample_rejections(
    samples,
    scores,
    threshold,
    subdivisions,
    character_risks=None,
    risk_threshold=None,
):
    """Вычисляет причины отклонения для списка размеченных записей"""

    results = []
    for index, ((record, _, _), score) in enumerate(zip(samples, scores)):
        fields = extract_features(record)[1]
        results.append(rejection_reasons(
            record,
            fields,
            score,
            threshold,
            subdivisions,
            character_risks[index] if character_risks is not None else None,
            risk_threshold,
        ))
    return results


def summarize_decisions(reasons_by_record, labels):
    """Подсчитывает принятые записи и ошибки по итоговым решениям"""

    accepted_ids = [
        index for index, reasons in enumerate(reasons_by_record)
        if not reasons
    ]
    error_count = sum(labels[index] == 0 for index in accepted_ids)
    accepted_count = len(accepted_ids)
    sample_count = len(labels)
    return {
        "n": sample_count,
        "accepted": accepted_count,
        "errors_among_accepted": error_count,
        "correct_accepts": accepted_count - error_count,
        "rejected": sample_count - accepted_count,
        "acceptance_rate": accepted_count / sample_count if sample_count else 0.0,
    }


def rejection_report_row(
    split,
    record,
    predicted_fields,
    expected_fields,
    field_scores,
    reasons,
    character_error_risk=None,
):
    """Собирает строку отчёта об отклонённой записи"""

    field_differences = {}
    if expected_fields is not None:
        for name in FIELDS:
            if normalize_text(predicted_fields[name]) != normalize_text(
                expected_fields.get(name)
            ):
                field_differences[name] = {
                    "predicted": predicted_fields[name],
                    "expected": expected_fields.get(name),
                }

    row = {
        "split": split,
        "doc_id": record["doc_id"],
        "rejection_reasons": reasons,
        "field_confidence": dict(zip(FIELDS, field_scores)),
        "minimum_field_confidence": min(field_scores),
        "minimum_character_confidence": minimum_character_confidence(record),
        "predicted": predicted_fields,
        "expected": expected_fields,
        "field_differences": field_differences,
    }
    if character_error_risk is not None:
        row["character_error_risk"] = character_error_risk
    return row


def write_answers(
    test_records,
    model_data,
    threshold,
    output_path,
    rejected_report_path,
):
    """Сохраняет ответы и причины отклонения записей test"""

    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Не найден справочник подразделений: {REFERENCE_PATH}"
        )

    # Справочник и правила валидности дополняют оценку точности OCR-модели.
    known_subdivisions = load_subdivisions(REFERENCE_PATH)
    extracted = [
        extract_features(record, known_subdivisions)
        for record in test_records
    ]
    feature_names = model_data["feature_names"]
    feature_rows = [
        [features[name] for name in feature_names]
        for features, _ in extracted
    ]
    scores = predict_many(model_data, feature_rows)
    character_risks = [
        document_character_error_risk(record, model_data["character_error_calibrator"])
        for record in test_records
    ]
    risk_threshold = float(model_data["character_error_risk_threshold"])

    with output_path.open("w", encoding="utf-8") as output_file:
        with rejected_report_path.open("a", encoding="utf-8") as report_file:
            for index, (record, (_, fields), score) in enumerate(zip(
                test_records,
                extracted,
                scores,
            )):
                reasons = rejection_reasons(
                    record,
                    fields,
                    score,
                    threshold,
                    known_subdivisions,
                    character_risks[index],
                    risk_threshold,
                )
                accepted = not reasons
                answer = {
                    "doc_id": record["doc_id"],
                    "decision": "accept" if accepted else "reject",
                }
                if accepted:
                    answer["fields"] = fields
                else:
                    report_file.write(json.dumps(
                        rejection_report_row(
                            "test",
                            record,
                            fields,
                            None,
                            score,
                            reasons,
                            character_risks[index],
                        ),
                        ensure_ascii=False,
                    ) + "\n")

                output_file.write(
                    json.dumps(answer, ensure_ascii=False) + "\n"
                )


def train_model():
    """Обучает модель, калибрует порог, оценивает выборки и создаёт ответы"""

    train_predictions = load_jsonl(ROOT / "data/train_predictions.jsonl")
    train_labels = load_jsonl(ROOT / "data/train_labels.jsonl")
    labels_by_id = {item["doc_id"]: item for item in train_labels}

    if len(train_predictions) != len(labels_by_id):
        raise ValueError("Число train_predictions и train_labels не совпадает")

    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Не найден справочник подразделений: {REFERENCE_PATH}"
        )
    known_subdivisions = load_subdivisions(REFERENCE_PATH)

    samples = []
    for record in train_predictions:
        doc_id = record["doc_id"]
        if doc_id not in labels_by_id:
            raise ValueError(f"Нет train_labels для {doc_id}")

        features, predicted_fields = extract_features(
            record,
            known_subdivisions,
        )
        targets = field_correctness_targets(
            predicted_fields,
            labels_by_id[doc_id],
        )
        samples.append((record, features, targets))

    # Одинаковый seed делает разбиение на части воспроизводимым.
    random.Random(SEED).shuffle(samples)
    sample_count = len(samples)

    fit_end = int(sample_count * 0.60)
    calibration_end = int(sample_count * 0.80)

    # Fit обучает веса, calibration выбирает порог, validation оценивает
    # выбранную стратегию на независимых примерах.
    fit_samples = samples[:fit_end]
    calibration_samples = samples[fit_end:calibration_end]
    validation_samples = samples[calibration_end:]

    feature_names = list(samples[0][1])
    fit_features = [
        [features[name] for name in feature_names]
        for _, features, _ in fit_samples
    ]
    fit_targets = [targets for _, _, targets in fit_samples]
    fit_labels = [int(all(targets)) for targets in fit_targets]

    # Подбираем оценку модели на калибровочной выборке
    calibration_features = [
        [features[name] for name in feature_names]
        for _, features, _ in calibration_samples
    ]
    calibration_targets = [targets for _, _, targets in calibration_samples]
    calibration_labels = [int(all(targets)) for targets in calibration_targets]

    # Подбираем базовую сеть и её настройки только по fit и calibration.
    model_data, char_confidence_scale, _ = select_char_confidence_scale(
        fit_features,
        fit_targets,
        feature_names,
        calibration_features,
        calibration_samples,
        calibration_targets,
        known_subdivisions,
    )
    calibration_scores = predict_many(model_data, calibration_features)

    # Калибратор символов обучается только на fit, не видя calibration/holdout.
    character_observations = []
    for record, _, _ in fit_samples:
        expected_fields = labels_by_id[record["doc_id"]]["fields"]
        character_observations.extend(
            character_observations_for_record(record, expected_fields)
        )
    character_calibrator = fit_character_error_calibrator(
        character_observations
    )
    model_data["character_error_calibrator"] = character_calibrator

    calibration_risks = [
        document_character_error_risk(record, character_calibrator)
        for record, _, _ in calibration_samples
    ]
    calibration_base_rejections = evaluate_sample_rejections(
        calibration_samples,
        calibration_scores,
        -1.0,
        known_subdivisions,
    )
    calibration_eligible = [
        not reasons for reasons in calibration_base_rejections
    ]
    threshold, risk_threshold = select_document_and_risk_thresholds(
        [min(row) for row in calibration_scores],
        calibration_labels,
        calibration_risks,
        calibration_eligible,
    )

    # Проверяем оценку модели на проверочной выборке
    validation_features = [
        [features[name] for name in feature_names]
        for _, features, _ in validation_samples
    ]
    validation_targets = [targets for _, _, targets in validation_samples]
    validation_labels = [int(all(targets)) for targets in validation_targets]
    validation_scores = predict_many(model_data, validation_features)
    validation_risks = [
        document_character_error_risk(record, character_calibrator)
        for record, _, _ in validation_samples
    ]
    # Теперь проверяем оценку на обучающей выборке
    fit_scores = predict_many(model_data, fit_features)
    fit_risks = [
        document_character_error_risk(record, character_calibrator)
        for record, _, _ in fit_samples
    ]

    # Оцениваем все фильтры, включая формат, валидность и справочник.
    fit_rejection_reasons = evaluate_sample_rejections(
        fit_samples,
        fit_scores,
        threshold,
        known_subdivisions,
        fit_risks,
        risk_threshold,
    )
    calibration_rejection_reasons = evaluate_sample_rejections(
        calibration_samples,
        calibration_scores,
        threshold,
        known_subdivisions,
        calibration_risks,
        risk_threshold,
    )
    validation_rejection_reasons = evaluate_sample_rejections(
        validation_samples,
        validation_scores,
        threshold,
        known_subdivisions,
        validation_risks,
        risk_threshold,
    )

    # Метрики учитывают те же причины отказа, что и итоговое решение.
    fit_metrics = summarize_decisions(fit_rejection_reasons, fit_labels)
    calibration_metrics = summarize_decisions(
        calibration_rejection_reasons,
        calibration_labels,
    )
    validation_metrics = summarize_decisions(
        validation_rejection_reasons,
        validation_labels,
    )

    # Сохраняем все отклонённые holdout и test в один JSONL-отчёт.
    rejected_report_path = ROOT / "solution/rejected_records.jsonl"
    with rejected_report_path.open("w", encoding="utf-8") as report_file:
        for index, ((record, _, _), score, reasons) in enumerate(zip(
            validation_samples,
            validation_scores,
            validation_rejection_reasons,
        )):
            if not reasons:
                continue
            predicted_fields = extract_features(record)[1]
            expected_fields = labels_by_id[record["doc_id"]]["fields"]
            report_file.write(json.dumps(
                rejection_report_row(
                    "holdout",
                    record,
                    predicted_fields,
                    expected_fields,
                    score,
                    reasons,
                    validation_risks[index],
                ),
                ensure_ascii=False,
            ) + "\n")

    # Печатаем принятые holdout-записи, в которых есть ошибка OCR.
    print("\nОшибочно принятые записи holdout:", flush=True)
    holdout_errors_path = ROOT / "solution/holdout_false_accepts.jsonl"
    printed_errors = 0
    with holdout_errors_path.open("w", encoding="utf-8") as errors_file:
        for index, ((record, _, target), score, reasons) in enumerate(zip(
            validation_samples,
            validation_scores,
            validation_rejection_reasons,
        )):
            if all(target) or reasons:
                continue

            predicted_fields = extract_features(record)[1]
            expected_fields = labels_by_id[record["doc_id"]]["fields"]
            errors = {
                name: {
                    "predicted": predicted_fields[name],
                    "expected": expected_fields[name],
                    "ocr_details": (record.get("viz") or {}).get(name),
                }
                for name in FIELDS
                if normalize_text(predicted_fields[name])
                != normalize_text(expected_fields[name])
            }
            if errors:
                error_row = {
                    "split": "holdout",
                    "doc_id": record["doc_id"],
                    "field_confidence": dict(zip(FIELDS, score)),
                    "character_error_risk": validation_risks[index],
                    "errors": errors,
                }
                errors_file.write(
                    json.dumps(error_row, ensure_ascii=False) + "\n"
                )
                print(json.dumps({
                    "doc_id": record["doc_id"],
                    "field_confidence": dict(zip(FIELDS, score)),
                    "errors": errors,
                }, ensure_ascii=False), flush=True)
                printed_errors += 1

    if printed_errors == 0:
        print("Нет ошибочно принятых записей в holdout.", flush=True)

    field_error_counts = {name: 0 for name in FIELDS}
    exact_document_count = 0
    for record, _, target in samples:
        exact_document_count += int(all(target))
        predicted_fields = extract_features(record)[1]
        expected_fields = labels_by_id[record["doc_id"]]["fields"]
        for name in FIELDS:
            predicted_value = normalize_text(predicted_fields[name])
            expected_value = normalize_text(expected_fields[name])
            if predicted_value != expected_value:
                field_error_counts[name] += 1

    model_data["document_threshold"] = threshold
    model_data["character_error_risk_threshold"] = risk_threshold
    model_data["target_document_precision"] = TARGET_DOCUMENT_PRECISION
    model_data["char_confidence_scale"] = char_confidence_scale
    model_data["minimum_organization_name_similarity"] = (
        MIN_ORGANIZATION_NAME_SIMILARITY
    )
    model_data["training_split"] = {
        "fit": len(fit_samples),
        "calibration": len(calibration_samples),
        "validation": len(validation_samples),
    }
    model_data["target"] = "per_field_exactness"

    # Сохраняем все параметры, нужные для воспроизведения предсказаний
    model_path = ROOT / "solution/calibrator.json"
    model_path.write_text(
        json.dumps(model_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Применяем модель и фильтры действительности к test-записям
    test_records = load_jsonl(ROOT / "data/test_predictions.jsonl")
    answers_path = ROOT / "solution/answers.jsonl"
    write_answers(
        test_records,
        model_data,
        threshold,
        answers_path,
        rejected_report_path,
    )

    with answers_path.open(encoding="utf-8") as answers_file:
        test_accept_count = sum(
            json.loads(line)["decision"] == "accept"
            for line in answers_file
        )

    summary = {
        "train_documents": sample_count,
        "all_10_fields_exact": exact_document_count,
        "all_10_fields_with_errors": sample_count - exact_document_count,
        "field_error_counts": field_error_counts,
        "calibrated_threshold": threshold,
        "character_error_risk_threshold": risk_threshold,
        "target_document_precision": TARGET_DOCUMENT_PRECISION,
        "char_confidence_scale": char_confidence_scale,
        "minimum_organization_name_similarity": (
            MIN_ORGANIZATION_NAME_SIMILARITY
        ),
        "fit": fit_metrics,
        "calibration": calibration_metrics,
        "holdout_validation": validation_metrics,
        "character_observations_fit": len(character_observations),
        "character_error_probability_at_0_9273": (
            character_error_probability(0.9273, character_calibrator)
        ),
        "holdout_false_accepts": str(holdout_errors_path),
        "test_documents_scored": len(test_records),
        "test_accepts": test_accept_count,
        "artifacts": {
            "model": str(model_path),
            "answers": str(answers_path),
            "rejected_records": str(rejected_report_path),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def calibrate_existing_model():
    """Обучает калибратор символов и пороги на сохранённых весах модели"""

    source_model_path = ROOT / "solution/calibrator.json"
    if not source_model_path.exists():
        raise FileNotFoundError(
            f"Не найдена исходная модель: {source_model_path}"
        )
    model_data = json.loads(source_model_path.read_text(encoding="utf-8"))
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Не найден справочник подразделений: {REFERENCE_PATH}"
        )

    train_predictions = load_jsonl(ROOT / "data/train_predictions.jsonl")
    train_labels = load_jsonl(ROOT / "data/train_labels.jsonl")
    labels_by_id = {item["doc_id"]: item for item in train_labels}
    known_subdivisions = load_subdivisions(REFERENCE_PATH)

    samples = []
    for record in train_predictions:
        label = labels_by_id.get(record["doc_id"])
        if label is None:
            raise ValueError(f"Нет train_labels для {record['doc_id']}")
        features, predicted_fields = extract_features(
            record,
            known_subdivisions,
        )
        targets = field_correctness_targets(predicted_fields, label)
        samples.append((record, features, targets))

    random.Random(SEED).shuffle(samples)
    fit_end = int(len(samples) * 0.60)
    calibration_end = int(len(samples) * 0.80)
    fit_samples = samples[:fit_end]
    calibration_samples = samples[fit_end:calibration_end]
    validation_samples = samples[calibration_end:]
    feature_names = model_data["feature_names"]

    def features_for(selected_samples):
        return [
            [features[name] for name in feature_names]
            for _, features, _ in selected_samples
        ]

    calibration_features = features_for(calibration_samples)
    calibration_scores = predict_many(model_data, calibration_features)
    calibration_labels = [
        int(all(targets)) for _, _, targets in calibration_samples
    ]
    character_observations = []
    for record, _, _ in fit_samples:
        character_observations.extend(character_observations_for_record(
            record,
            labels_by_id[record["doc_id"]]["fields"],
        ))
    character_calibrator = fit_character_error_calibrator(
        character_observations
    )
    model_data["character_error_calibrator"] = character_calibrator
    calibration_risks = [
        document_character_error_risk(record, character_calibrator)
        for record, _, _ in calibration_samples
    ]
    eligible_rejections = evaluate_sample_rejections(
        calibration_samples,
        calibration_scores,
        -1.0,
        known_subdivisions,
    )
    eligible = [not reasons for reasons in eligible_rejections]
    document_threshold, risk_threshold = select_document_and_risk_thresholds(
        [min(row) for row in calibration_scores],
        calibration_labels,
        calibration_risks,
        eligible,
    )

    model_data["document_threshold"] = document_threshold
    model_data["character_error_risk_threshold"] = risk_threshold
    model_data["target_document_precision"] = TARGET_DOCUMENT_PRECISION
    model_data["calibration_method"] = "existing_weights_plus_character_error_risk"

    validation_scores = predict_many(model_data, features_for(validation_samples))
    validation_risks = [
        document_character_error_risk(record, character_calibrator)
        for record, _, _ in validation_samples
    ]
    validation_labels = [
        int(all(targets)) for _, _, targets in validation_samples
    ]
    calibration_reasons = evaluate_sample_rejections(
        calibration_samples,
        calibration_scores,
        document_threshold,
        known_subdivisions,
        calibration_risks,
        risk_threshold,
    )
    validation_reasons = evaluate_sample_rejections(
        validation_samples,
        validation_scores,
        document_threshold,
        known_subdivisions,
        validation_risks,
        risk_threshold,
    )
    calibration_metrics = summarize_decisions(
        calibration_reasons,
        calibration_labels,
    )
    validation_metrics = summarize_decisions(
        validation_reasons,
        validation_labels,
    )

    rejected_report_path = ROOT / "solution/rejected_records.jsonl"
    holdout_errors_path = ROOT / "solution/holdout_false_accepts.jsonl"
    with rejected_report_path.open("w", encoding="utf-8") as report_file:
        for index, ((record, _, _), score, reasons) in enumerate(zip(
            validation_samples,
            validation_scores,
            validation_reasons,
        )):
            if reasons:
                expected = labels_by_id[record["doc_id"]]["fields"]
                predicted = extract_features(record)[1]
                report_file.write(json.dumps(
                    rejection_report_row(
                        "holdout",
                        record,
                        predicted,
                        expected,
                        score,
                        reasons,
                        validation_risks[index],
                    ),
                    ensure_ascii=False,
                ) + "\n")

    with holdout_errors_path.open("w", encoding="utf-8") as errors_file:
        for index, ((record, _, targets), score, reasons) in enumerate(zip(
            validation_samples,
            validation_scores,
            validation_reasons,
        )):
            if all(targets) or reasons:
                continue
            predicted = extract_features(record)[1]
            expected = labels_by_id[record["doc_id"]]["fields"]
            errors = {
                name: {
                    "predicted": predicted[name],
                    "expected": expected[name],
                    "ocr_details": (record.get("viz") or {}).get(name),
                }
                for name in FIELDS
                if normalize_text(predicted[name])
                != normalize_text(expected.get(name))
            }
            errors_file.write(json.dumps({
                "split": "holdout",
                "doc_id": record["doc_id"],
                "field_confidence": dict(zip(FIELDS, score)),
                "character_error_risk": validation_risks[index],
                "errors": errors,
            }, ensure_ascii=False) + "\n")

    model_path = ROOT / "solution/calibrator.json"
    model_path.write_text(
        json.dumps(model_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    answers_path = ROOT / "solution/answers.jsonl"
    test_records = load_jsonl(ROOT / "data/test_predictions.jsonl")
    write_answers(
        test_records,
        model_data,
        document_threshold,
        answers_path,
        rejected_report_path,
    )
    answers = load_jsonl(answers_path)

    print(json.dumps({
        "calibration_method": "existing_saved_weights",
        "document_threshold": document_threshold,
        "character_error_risk_threshold": risk_threshold,
        "character_observations_fit": len(character_observations),
        "character_error_probability_at_0_9273": (
            character_error_probability(0.9273, character_calibrator)
        ),
        "calibration": calibration_metrics,
        "holdout_validation": validation_metrics,
        "holdout_false_accepts": str(holdout_errors_path),
        "test_documents_scored": len(test_records),
        "test_accepts": sum(
            answer["decision"] == "accept" for answer in answers
        ),
        "artifacts": {
            "model": str(model_path),
            "answers": str(answers_path),
            "rejected_records": str(rejected_report_path),
        },
    }, ensure_ascii=False, indent=2))


def main(argv):
    """Обрабатывает входной JSONL сохранённой моделью или запускает обучение"""

    if len(argv) == 2 and argv[1] == "--train":
        train_model()
        return
    if len(argv) == 2 and argv[1] == "--calibrate-risk":
        calibrate_existing_model()
        return

    if len(argv) != 3:
        raise SystemExit(
            "Запуск обработки: python3 solution/postprocess.py "
            "ВХОДНОЙ.jsonl ВЫХОДНОЙ.jsonl\n"
            "Переобучение модели: python3 solution/postprocess.py --train\n"
            "Калибровка риска на сохранённых весах: "
            "python3 solution/postprocess.py --calibrate-risk"
        )

    input_path = Path(argv[1])
    output_path = Path(argv[2])
    if not input_path.exists():
        raise FileNotFoundError(f"Не найден входной файл: {input_path}")
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Не найден справочник подразделений: {REFERENCE_PATH}"
        )

    model_path = ROOT / "solution/calibrator.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Не найдена обученная модель: {model_path}; "
            "сначала запустите python3 solution/postprocess.py --train"
        )

    model_data = json.loads(model_path.read_text(encoding="utf-8"))
    threshold = float(model_data["document_threshold"])
    rejected_report_path = ROOT / "solution/rejected_records.jsonl"
    rejected_report_path.write_text("", encoding="utf-8")

    records = load_jsonl(input_path)
    write_answers(
        records,
        model_data,
        threshold,
        output_path,
        rejected_report_path,
    )

    answers = load_jsonl(output_path)
    accepted = sum(answer["decision"] == "accept" for answer in answers)
    rejected = len(answers) - accepted
    print(f"Принято паспортов: {accepted}")
    print(f"Отклонено паспортов: {rejected}")
    print(
        "Порог вероятности ошибки символа в документе: "
        f"{model_data['character_error_risk_threshold']:.6f}"
    )


if __name__ == "__main__":
    main(sys.argv)
