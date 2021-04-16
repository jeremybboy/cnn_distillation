# import numpy as np
import os
import torchvision.datasets
from torchvision import transforms
import torch
from models.cnn_teacher import NeuralNet
from models.cnn_student import StudentNet
from training_classif import ClassifTrainer
from training_distill import DistillationTrainer
# from training_distill_nofeature import DistillationTrainer
from torch.utils.tensorboard import SummaryWriter


#Define device (cuda if it exists)
# moves your models to train on your gpu if available else it uses your cpu
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#Prepare data
batch_size = 64
# data_path ="C:/Users/Jeremy UZAN/data"
data_path ="/home/jeremy/data"
train_set = torchvision.datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
trainLoader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
testset = torchvision.datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)


print("DATA IS LOADED")

####################### I. TRAIN TEACHER #######################

#Create teacher model
teacher = NeuralNet()
teacher.to(device)
print("MODEL IS REA DY")

num_params = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
print('Number of parameters of the teacher: %d' % num_params)

#Create TeacherTrainer
teacher_dir = "results/teacher51"
os.makedirs(teacher_dir, exist_ok=True)
teacher_trainer = ClassifTrainer(teacher_dir, teacher, trainLoader, testloader, lr=0.0001)
print("TRAINER IS READY")

#train teacher
print("TRAINING MODEL")
teacher_trainer.train_model(80, device)

#Save teacher
trained_dir="./trained/teacher2/"
os.makedirs(trained_dir, exist_ok=True)
torch.save(teacher.state_dict(), './trained/teacher2/TrainedTeacher.pt')
print("trained teacher saved")


# # ####################### II. TRAIN SMALL MODEL #######################
# #
# # #Create Student model
# small = StudentNet()
# small.to(device)
# print("MODEL IS READY")
#
# num_params_student = sum(p.numel() for p in small.parameters() if p.requires_grad)
# print('Number of parameters of the student: %d' % num_params_student)
#
# #Create StudentTrainer
# student_dir="./results/student50"
# os.makedirs(student_dir, exist_ok=True)
# small_trainer = ClassifTrainer(student_dir, small, trainLoader, testloader, lr=0.0001)
# print("TRAINER IS READY")
# #
# #train student
# print("TRAINING MDEL")
# small_trainer.train_model(1, device)

teacher=NeuralNet()
teacher.to(device)
teacher.load_state_dict(torch.load('./trained/teacher2/TrainedTeacher.pt', map_location="cuda:0"))


#
# #Create Distill model
# distill = StudentNet()
# distill.to(device)
# print("MODEL Distill IS READY")
# #
# #Create DistillTrainer
# distill_dir="./results/distill49temp1"
# os.makedirs(distill_dir, exist_ok=True)
# distill_trainer = DistillationTrainer(distill_dir, distill, teacher, trainLoader, testloader, lr=0.001)
# print("TRAINER IS READY")
#
# #train student
# print("TRAINING distilled MODEL")
# distill_trainer.train_model(2, device)