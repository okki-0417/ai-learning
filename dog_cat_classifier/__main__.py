import argparse

from .model import load_model
from .preprocess import preprocess_image
from .classifier import classify, classify_all
from .presenter import print_result, print_top_n_result


def main():
    parser = argparse.ArgumentParser(description="犬猫画像分類器")
    parser.add_argument("image_path", help="画像ファイルパス")
    parser.add_argument("--all", "-a", action="store_true", help="全クラスからtop N予測を表示")
    parser.add_argument("--top", "-n", type=int, default=5, help="表示する予測数 (default: 5)")
    args = parser.parse_args()

    model, categories = load_model()
    image_tensor = preprocess_image(args.image_path)

    if args.all:
        result = classify_all(model, image_tensor, categories, top_n=args.top)
        print_top_n_result(result)
    else:
        result = classify(model, image_tensor, categories)
        print_result(result)


if __name__ == "__main__":
    main()
