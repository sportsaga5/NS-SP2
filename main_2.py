import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns


# 1. КОНФИГУРАЦИЯ
class Config:
    seed = 42
    batch_size = 128
    latent_dim = 100
    num_classes = 10  # EMNIST Digits: 10 цифр 0-9
    img_shape = (1, 28, 28)
    lr = 0.0002
    betas = (0.5, 0.999)
    n_epochs = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples_dir = "generated_samples_emnist_digits"
    models_dir = "saved_models_emnist_digits"
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)


config = Config()
torch.manual_seed(config.seed)
np.random.seed(config.seed)

print(f"Device: {config.device}")
print(f"Batch size: {config.batch_size}")
print(f"Latent dim: {config.latent_dim}")
print(f"Number of classes (EMNIST Digits 0-9): {config.num_classes}")


# 2. ЗАГРУЗКА ДАННЫХ (EMNIST Digits)
def load_data():
    """Загрузка EMNIST Digits, нормализация в [-1, 1]"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # EMNIST Digits: split="digits" — цифры 0-9
    # ВНИМАНИЕ: EMNIST Digits возвращает метки 0-9, преобразование НЕ нужно!
    train_dataset = datasets.EMNIST(
        root="./data",
        split="digits",      # ← ключевое отличие: цифры
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.EMNIST(
        root="./data",
        split="digits",
        train=False,
        download=True,
        transform=transform
    )

    # EMNIST Digits возвращает метки 0-9 (в отличие от Letters, где 1-26)
    # Поэтому НЕ нужен LabelTransformWrapper!

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print(f"Classes: 0, 1, 2, ..., 9 (0-9)")
    return train_loader, test_loader


# 3. МОДЕЛИ
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_shape):
        super().__init__()
        self.img_shape = img_shape
        self.label_embedding = nn.Embedding(num_classes, latent_dim)

        self.model = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_embed = self.label_embedding(labels)
        inputs = torch.cat([z, label_embed], dim=1)
        img = self.model(inputs)
        img = img.view(-1, *self.img_shape)
        return img


class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes, img_shape):
        super().__init__()
        self.img_shape = img_shape
        self.label_embedding = nn.Embedding(num_classes, int(np.prod(img_shape)))

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)) * 2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        label_embed = self.label_embedding(labels)
        inputs = torch.cat([img_flat, label_embed], dim=1)
        validity = self.model(inputs)
        return validity


# 4. ОБУЧЕНИЕ cGAN
def train_cgan(train_loader, test_loader):
    generator = ConditionalGenerator(
        config.latent_dim, config.num_classes, config.img_shape
    ).to(config.device)

    discriminator = ConditionalDiscriminator(
        config.num_classes, config.img_shape
    ).to(config.device)

    g_optimizer = optim.Adam(generator.parameters(), lr=config.lr, betas=config.betas)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=config.lr, betas=config.betas)

    criterion = nn.BCELoss()

    history = {
        "g_loss": [], "d_loss": [],
        "real_acc": [], "fake_acc": []
    }

    fixed_noise = torch.randn(64, config.latent_dim, device=config.device)
    fixed_labels = torch.randint(0, config.num_classes, (64,), device=config.device)

    print("\n" + "=" * 60)
    print("Начало обучения Conditional GAN на EMNIST Digits (0-9)")
    print("=" * 60 + "\n")

    for epoch in range(1, config.n_epochs + 1):
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_real_acc = 0
        epoch_fake_acc = 0

        for batch_idx, (real_imgs, labels) in enumerate(train_loader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(config.device)
            labels = labels.to(config.device)  # уже 0-9

            real_labels = torch.ones(batch_size, 1, device=config.device)
            fake_labels = torch.zeros(batch_size, 1, device=config.device)

            # --- Обучение дискриминатора ---
            d_optimizer.zero_grad()

            real_validity = discriminator(real_imgs, labels)
            d_real_loss = criterion(real_validity, real_labels)

            z = torch.randn(batch_size, config.latent_dim, device=config.device)
            fake_imgs = generator(z, labels)
            fake_validity = discriminator(fake_imgs.detach(), labels)
            d_fake_loss = criterion(fake_validity, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # --- Обучение генератора ---
            g_optimizer.zero_grad()

            z = torch.randn(batch_size, config.latent_dim, device=config.device)
            fake_imgs = generator(z, labels)
            fake_validity = discriminator(fake_imgs, labels)
            g_loss = criterion(fake_validity, real_labels)

            g_loss.backward()
            g_optimizer.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_real_acc += (real_validity > 0.5).float().mean().item()
            epoch_fake_acc += (fake_validity < 0.5).float().mean().item()

        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / len(train_loader)
        avg_real_acc = epoch_real_acc / len(train_loader)
        avg_fake_acc = epoch_fake_acc / len(train_loader)

        history["g_loss"].append(avg_g_loss)
        history["d_loss"].append(avg_d_loss)
        history["real_acc"].append(avg_real_acc)
        history["fake_acc"].append(avg_fake_acc)

        print(f"[Эпоха {epoch}/{config.n_epochs}] "
            f"D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}, "
            f"Real_acc: {avg_real_acc:.3f}, Fake_acc: {avg_fake_acc:.3f}")

        if epoch % 10 == 0 or epoch == 1:
            generate_and_save_samples(generator, fixed_noise, fixed_labels, epoch)

    torch.save(generator.state_dict(), f"{config.models_dir}/generator_final.pth")
    torch.save(discriminator.state_dict(), f"{config.models_dir}/discriminator_final.pth")

    return generator, discriminator, history


# 5. ВИЗУАЛИЗАЦИЯ
def get_class_name(label_id):
    return str(label_id)  # просто цифра


def generate_and_save_samples(generator, noise, labels, epoch):
    generator.eval()
    with torch.no_grad():
        fake_imgs = generator(noise, labels)
        fake_imgs = (fake_imgs + 1) / 2

        grid = utils.make_grid(fake_imgs, nrow=8, padding=2, normalize=False)
        plt.figure(figsize=(12, 12))
        plt.imshow(grid.cpu().permute(1, 2, 0).numpy(), cmap='gray')
        plt.axis('off')
        plt.title(f"Сгенерированные цифры (EMNIST Digits), эпоха {epoch}")
        plt.savefig(f"{config.samples_dir}/epoch_{epoch}.png", bbox_inches='tight')
        plt.close()
    generator.train()


def plot_losses(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["g_loss"], label="Generator loss")
    plt.plot(history["d_loss"], label="Discriminator loss")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.title("Динамика лоссов (EMNIST Digits)")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history["real_acc"], label="Real accuracy")
    plt.plot(history["fake_acc"], label="Fake accuracy")
    plt.xlabel("Эпоха")
    plt.ylabel("Accuracy")
    plt.title("Точность дискриминатора (EMNIST Digits)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_history_emnist_digits.png")
    plt.show()


def generate_controlled_samples(generator, num_samples=64, fixed_class=None):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, config.latent_dim, device=config.device)

        if fixed_class is not None:
            labels = torch.full((num_samples,), fixed_class, device=config.device, dtype=torch.long)
            class_name = get_class_name(fixed_class)
        else:
            labels = torch.randint(0, config.num_classes, (num_samples,), device=config.device)
            class_name = "всех классов"

        fake_imgs = generator(noise, labels)
        fake_imgs = (fake_imgs + 1) / 2

        grid = utils.make_grid(fake_imgs, nrow=8, padding=2)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.cpu().permute(1, 2, 0).numpy(), cmap='gray')
        plt.axis('off')
        plt.title(f"Сгенерированные цифры {class_name}")

        first_labels = [get_class_name(l.item()) for l in labels[:8]]
        print(f"Сгенерированы цифры: {first_labels}...")
        plt.show()
    generator.train()


def style_interpolation(generator, class_id, num_steps=10):
    generator.eval()
    with torch.no_grad():
        z1 = torch.randn(1, config.latent_dim, device=config.device)
        z2 = torch.randn(1, config.latent_dim, device=config.device)

        labels = torch.full((num_steps,), class_id, device=config.device, dtype=torch.long)
        class_name = get_class_name(class_id)

        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        axes = axes.flatten()

        for i, alpha in enumerate(np.linspace(0, 1, num_steps)):
            z = (1 - alpha) * z1 + alpha * z2
            img = generator(z, labels[i:i + 1])
            img = (img + 1) / 2

            axes[i].imshow(img.squeeze().cpu().numpy(), cmap='gray')
            axes[i].set_title(f"α={alpha:.1f}")
            axes[i].axis('off')

        plt.suptitle(f"Интерполяция стиля для цифры {class_name}")
        plt.tight_layout()
        plt.savefig("style_interpolation_emnist_digits.png")
        plt.show()
    generator.train()


# 6. ОЦЕНКА КАЧЕСТВА
def evaluate_with_classifier(generator, test_loader):
    print("\n" + "=" * 60)
    print("Оценка качества: обучение классификатора на сгенерированных цифрах")
    print("=" * 60)

    class SimpleClassifier(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout = nn.Dropout2d(0.25)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.conv1(x)
            x = nn.functional.relu(x)
            x = self.conv2(x)
            x = nn.functional.relu(x)
            x = nn.functional.max_pool2d(x, 2)
            x = self.dropout(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = nn.functional.relu(x)
            x = self.fc2(x)
            return nn.functional.log_softmax(x, dim=1)

    generator.eval()
    num_gen_samples = 10000
    gen_imgs = []
    gen_labels = []

    with torch.no_grad():
        for _ in range(num_gen_samples // config.batch_size):
            z = torch.randn(config.batch_size, config.latent_dim, device=config.device)
            labels = torch.randint(0, config.num_classes, (config.batch_size,), device=config.device)
            imgs = generator(z, labels)
            gen_imgs.append(imgs.cpu())
            gen_labels.append(labels.cpu())

    X_gen = torch.cat(gen_imgs, dim=0)
    y_gen = torch.cat(gen_labels, dim=0)

    real_imgs = []
    real_labels = []
    for imgs, labels in test_loader:
        real_imgs.append(imgs)
        real_labels.append(labels)
        if len(torch.cat(real_imgs, dim=0)) >= 5000:
            break

    X_real = torch.cat(real_imgs, dim=0)[:5000]
    y_real = torch.cat(real_labels, dim=0)[:5000]

    classifier = SimpleClassifier(num_classes=config.num_classes).to(config.device)
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    dataset = torch.utils.data.TensorDataset(X_gen, y_gen)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    n_epochs = 5
    for epoch in range(n_epochs):
        classifier.train()
        for imgs, labels in loader:
            imgs, labels = imgs.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            output = classifier(imgs)
            loss = nn.functional.nll_loss(output, labels)
            loss.backward()
            optimizer.step()

    classifier.eval()
    with torch.no_grad():
        X_real = X_real.to(config.device)
        y_real = y_real.to(config.device)
        output = classifier(X_real)
        pred = output.argmax(dim=1)
        accuracy = accuracy_score(y_real.cpu(), pred.cpu())

    print(f"Точность классификатора (обучен на сгенерированных, тест на реальных): {accuracy:.4f}")

    cm = confusion_matrix(y_real.cpu(), pred.cpu())
    plt.figure(figsize=(10, 8))

    tick_labels = [str(i) for i in range(10)]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=tick_labels, yticklabels=tick_labels)
    plt.title("Confusion Matrix: Классификатор на сгенерированных цифрах")
    plt.xlabel("Предсказано")
    plt.ylabel("Истинное")
    plt.tight_layout()
    plt.savefig("confusion_matrix_emnist_digits.png")
    plt.show()

    return accuracy


# 7. ЗАПУСК
def main():
    print("=" * 60)
    print("ПРОЕКТ: ГЕНЕРАЦИЯ ЦИФР С КОНТРОЛЕМ СТИЛЯ (EMNIST Digits)")
    print("=" * 60)

    train_loader, test_loader = load_data()
    generator, discriminator, history = train_cgan(train_loader, test_loader)
    plot_losses(history)

    generate_controlled_samples(generator, num_samples=64, fixed_class=None)
    generate_controlled_samples(generator, num_samples=64, fixed_class=3)  # цифра 3
    generate_controlled_samples(generator, num_samples=64, fixed_class=7)  # цифра 7

    style_interpolation(generator, class_id=3, num_steps=10)  # цифра 3
    style_interpolation(generator, class_id=7, num_steps=10)  # цифра 7

    accuracy = evaluate_with_classifier(generator, test_loader)

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ПРОЕКТА")
    print("=" * 60)
    print(f"✅ Генератор обучен за {config.n_epochs} эпох")
    print(f"✅ Датасет: EMNIST Digits (280k цифр 0-9)")
    print(f"✅ Точность классификатора: {accuracy:.4f}")
    print(f"✅ Примеры сохранены в {config.samples_dir}")
    print(f"✅ Модели сохранены в {config.models_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()