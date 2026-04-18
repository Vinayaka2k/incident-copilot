import torch

from train_before_hyperparamchage import CNN


def count_parameters(model):
    """
    Count total and trainable parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def parameter_breakdown(model):
    """
    Print parameter count for each named parameter tensor.
    Also aggregate counts by top-level component such as
    features and classifier.
    """
    print("=== Per-parameter breakdown ===")
    component_counts = {}

    for name, param in model.named_parameters():
        num_params = param.numel()
        print(f"{name:40s} {list(param.shape)!s:25s} {num_params:>12,d}")

        top_component = name.split(".")[0]
        component_counts[top_component] = component_counts.get(top_component, 0) + num_params

    print("\n=== Top-level component breakdown ===")
    total = sum(component_counts.values())
    for component, count in component_counts.items():
        percent = 100.0 * count / total
        print(f"{component:20s} {count:>12,d} params   ({percent:6.2f}%)")


def layerwise_module_breakdown(model):
    """
    Print parameter count for immediate child modules.
    For your model this is mainly:
    - features
    - classifier
    """
    print("\n=== Immediate module breakdown ===")
    total = 0
    module_counts = {}

    for name, module in model.named_children():
        count = sum(p.numel() for p in module.parameters())
        module_counts[name] = count
        total += count

    for name, count in module_counts.items():
        percent = 100.0 * count / total if total > 0 else 0.0
        print(f"{name:20s} {count:>12,d} params   ({percent:6.2f}%)")


def largest_parameter_tensors(model, top_k=10):
    """
    Show the largest parameter tensors in the model.
    This helps identify which layers dominate parameter count.
    """
    params = []
    for name, param in model.named_parameters():
        params.append((name, param.numel(), list(param.shape)))

    params.sort(key=lambda x: x[1], reverse=True)

    print(f"\n=== Top {top_k} largest parameter tensors ===")
    for name, count, shape in params[:top_k]:
        print(f"{name:40s} {str(shape):25s} {count:>12,d}")


def analyze_fc_vs_conv(model):
    """
    Compare how many parameters are in Conv layers vs Linear layers.
    """
    conv_params = 0
    linear_params = 0
    other_params = 0

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_params += sum(p.numel() for p in module.parameters())
        elif isinstance(module, torch.nn.Linear):
            linear_params += sum(p.numel() for p in module.parameters())

    total_params = sum(p.numel() for p in model.parameters())
    other_params = total_params - conv_params - linear_params

    print("\n=== Conv vs Linear parameter analysis ===")
    print(f"Conv2d params : {conv_params:,}")
    print(f"Linear params : {linear_params:,}")
    print(f"Other params  : {other_params:,}")
    print(f"Total params  : {total_params:,}")

    if total_params > 0:
        print(f"Conv2d %      : {100.0 * conv_params / total_params:.2f}%")
        print(f"Linear %      : {100.0 * linear_params / total_params:.2f}%")
        print(f"Other %       : {100.0 * other_params / total_params:.2f}%")


def print_analysis_notes():
    """
    Prints short interpretation guidance for the homework discussion.
    """
    print("\n=== Analysis notes ===")
    print("1. If most parameters are in the classifier, especially the first Linear layer,")
    print("   then the fully connected part is dominating the model size.")
    print("2. In your CNN, the first classifier layer often holds the most parameters because")
    print("   it maps 256*14*14 features into 512 hidden units.")
    print("3. If training accuracy is high but validation accuracy is low, a very large")
    print("   classifier may be contributing to overfitting.")
    print("4. If both train and validation accuracy are low, the issue may be undertraining,")
    print("   optimizer settings, insufficient augmentation, or model design.")
    print("5. A common improvement is to reduce the flattened feature size using an")
    print("   AdaptiveAvgPool2d layer before the Linear layers.")


def main():
    model = CNN(num_classes=4)  # change num_classes if needed

    total_params, trainable_params = count_parameters(model)

    print("=== Overall parameter count ===")
    print(f"Total parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    parameter_breakdown(model)
    layerwise_module_breakdown(model)
    largest_parameter_tensors(model, top_k=10)
    analyze_fc_vs_conv(model)
    print_analysis_notes()


if __name__ == "__main__":
    main()