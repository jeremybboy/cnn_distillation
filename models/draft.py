def train_student():
    optimizer = optim.SGD(student_model.parameters(), lr=0.001)
    loss_classif = nn.CrossEntropyLoss()

    softmax_op = nn.Softmax(dim=1)
    mseloss_fn = nn.MSELoss()
    total_step = len(trainLoader)

    def logit_distillation_loss(logits_student, logits_teacher, T=5):

        loss = mseloss_fn(softmax_op(logits_student / T), softmax_op(logits_teacher / T))
        return loss

    def features_distillation_loss(fA, fB, f2, f4):

        loss = mseloss_fn(fA, f2) + mseloss_fn(fB, f4)
        return loss

    for epoch in range(num_epochs):
        train_loss1.append(0)
        train_loss2.append(0)
        train_loss3.append(0)
        for i, (images, labels) in enumerate(trainLoader):
            images = images.to(device)
            labels = labels.to(device)

            logits_student, fA, fB = student_model(images)

            # Forward pass
            logits_teacher, f2, f4 = model(images)

            loss1 = logit_distillation_loss(logits_student, logits_teacher, T=4)
            loss2 = loss_classif(logits_student, labels)
            loss3 = features_distillation_loss(fA, fB, f2, f4)

            loss = loss1 + loss2 + loss3
            # loss=loss2

            # predicted_labels=old scores
            # predicted_logits=predicted_labels
            # logits_student = ol predicted_logits
            # logits_teacher= old targets

            # Backprpagation and optimization
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            train_loss1[-1] += loss1.item()
            train_loss2[-1] += loss2.item()
            train_loss3[-1] += loss3.item()

            if (i + 1) % 100 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}, LC: {loss2.item():.4f}, LDl: {loss1.item():.4f}, LDf: {loss3.item():.4f}')

        test_loss1.append(0)
        test_loss2.append(0)
        test_loss3.append(0)

        with torch.no_grad():
            correct = 0
            total = 0
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            logits_student, fA, fB = student_model(images)
            logits_teacher, f2, f4 = model(images)

            _, predicted = torch.max(logits_student.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss1 = logit_distillation_loss(logits_student, logits_teacher, T=4)
            loss2 = loss_classif(logits_student, labels)
            loss3 = features_distillation_loss(fA, fB, f2, f4)
            loss = loss1 + loss2 + loss3

            test_loss1[-1] += loss1.item()
            test_loss2[-1] += loss2.item()
            test_loss3[-1] += loss3.item()

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], LossVal: {loss.item():.4f}, LCVal: {loss2.item():.4f}, LDlVal: {loss1.item():.4f}, LDfVal: {loss3.item():.4f}')


def teacher_test():
    # Test the models
    # In the test phase, don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs, f2, f4 = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy_teacher = (100 * correct / total)
    return accuracy_teacher


def student_test():
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs, fA, fB = student_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy_student = (100 * correct / total)
    return accuracy_student


def visual():
    import matplotlib.pyplot as plt
    # import numpy as np
    plt.ion()

    fig = plt.figure()
    plt.plot(train_loss)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


if __name__ == '__main__':
    train_teacher()
    print("Loading models successful!")
    accuracy_teacher = teacher_test()
    print(f"\nAccuracy of the network on the test images: {accuracy_teacher:.2f}%.\n")
    visual()
    train_student()
    print("Loadinf models successful")
    accuracy_student = student_test()
    print(f"\nAccuracy of the network on the test images: {accuracy_student:.2f}%.\n")
