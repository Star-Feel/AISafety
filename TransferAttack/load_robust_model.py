from robustbench.utils import load_model_from_ckpt, load_model

model = load_model(model_name='Bartoldson2024Adversarial_WRN-94-16',threat_model='Linf', dataset='cifar10')

model = load_model(model_name='Debenedetti2022Light_XCiT-L12', dataset='cifar10', threat_model='Linf')

model = load_model(model_name='Bai2023Improving_edm', dataset='cifar10', threat_model='Linf')

# model = load_model_from_ckpt(model_name='Bartoldson2024Adversarial_WRN-94-16',threat_model='Linf', dataset='cifar10', ckpt_path="./checkpoints/cifar10/Bartoldson2024Adversarial_WRN-94-16.pt")

# model = load_model_from_ckpt(model_name='Debenedetti2022Light_XCiT-L12', dataset='cifar10', threat_model='Linf', ckpt_path="./checkpoints/cifar10/debenedetti2022light-xcit-l-cifar10-linf.pth.tar")

# model = load_model_from_ckpt(model_name='Bai2023Improving_edm', dataset='cifar10', threat_model='Linf', ckpt_path="./checkpoints/cifar10/bit_rn-152-2.pt")