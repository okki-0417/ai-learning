import torch

from .model import DOG_CLASS_RANGE, CAT_CLASS_RANGE
from .schemas import Prediction, BreedPrediction, ClassificationResult, TopNResult


def get_top_breed(probabilities: torch.Tensor, categories: list, class_range: range) -> BreedPrediction:
    breed_probs = [(categories[i], probabilities[i].item()) for i in class_range]
    top = max(breed_probs, key=lambda x: x[1])
    return BreedPrediction(name=top[0], probability=top[1])


def get_total_probability(probabilities: torch.Tensor, class_range: range) -> float:
    return sum(probabilities[i].item() for i in class_range)


def classify(model, image_tensor: torch.Tensor, categories: list) -> ClassificationResult:
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    return ClassificationResult(
        dog_probability=get_total_probability(probabilities, DOG_CLASS_RANGE),
        cat_probability=get_total_probability(probabilities, CAT_CLASS_RANGE),
        top_dog_breed=get_top_breed(probabilities, categories, DOG_CLASS_RANGE),
        top_cat_breed=get_top_breed(probabilities, categories, CAT_CLASS_RANGE),
    )


def classify_all(model: torch.nn.Module, image_tensor: torch.Tensor, categories: list, top_n: int = 5) -> TopNResult:
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    all_predictions = [
        (categories[i], probabilities[i].item())
        for i in range(len(categories))
    ]
    sorted_predictions = sorted(all_predictions, key=lambda x: x[1], reverse=True)

    top_predictions = [
        Prediction(name=name, probability=prob)
        for name, prob in sorted_predictions[:top_n]
    ]

    return TopNResult(predictions=top_predictions)
