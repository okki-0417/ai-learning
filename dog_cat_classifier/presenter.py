from .schemas import ClassificationResult, TopNResult


def print_top_n_result(result: TopNResult) -> None:
    print("\n=== 予測結果 ===")
    for i, pred in enumerate(result.predictions, 1):
        print(f"{i}. {pred.name}: {pred.probability * 100:.1f}%")


def print_result(result: ClassificationResult) -> None:
    print("\n=== 判定結果 ===")
    print(f"犬の確率: {result.dog_probability * 100:.1f}%")
    print(f"猫の確率: {result.cat_probability * 100:.1f}%")

    if result.is_dog():
        breed = result.top_dog_breed
        print(f"\n判定: 犬 ({breed.name})")
        print(f"  犬種の確信度: {breed.probability * 100:.1f}%")
    elif result.is_cat():
        breed = result.top_cat_breed
        print(f"\n判定: 猫 ({breed.name})")
        print(f"  猫種の確信度: {breed.probability * 100:.1f}%")
    else:
        print("\n判定: どちらとも言えません")
