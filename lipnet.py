import torch
from typing import List
import os
import numpy as np
import cv2
import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset,random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from tqdm import tqdm
from torch.optim import Adam
from accelerate import (Accelerator, DistributedDataParallelKwargs,
                        DistributedType)

import wandb
from ctcdecode import CTCBeamDecoder
from accelerate.tracking import WandBTracker
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from datetime import timedelta
from contextlib import contextmanager, nullcontext


DEFAULT_DDP_KWARGS = DistributedDataParallelKwargs(find_unused_parameters = True)

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log
class CustomDataset(Dataset): 
    def __init__(self):
        
        # self.video_list = glob.glob('/dataset/grid/**/*.mpg', recursive=True)
        # self.video_list=sorted(self.video_list)
        # self.align_list = glob.glob('/dataset/grid/alignments/**/*.align',recursive=True)
        # self.align_list=sorted(self.align_list)
        
        self.video_list = glob.glob('/dataset/grid/s1/*.mpg')
        self.video_list=sorted(self.video_list)
        self.align_list = glob.glob('/dataset/grid/alignments/s1/*.align')
        self.align_list=sorted(self.align_list)
        
        assert len(self.align_list)==len(self.video_list), f"{len(self.align_list)}, {len(self.video_list)} 둘의 길이가 일치하지 않음."
        
        '''ensure that everthing is well aligned'''
        for i, (video_path, align_path) in enumerate(zip(self.video_list, self.align_list)):
            # os.path를 사용하여 파일 이름 추출
            video = os.path.splitext(os.path.basename(video_path))[0]
            align = os.path.splitext(os.path.basename(align_path))[0]
            if video != align:
                print(i,video_path, align_path)
        print("done loading the dataset")
        
    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, idx):
        frames=self.load_video(idx) 
        alignments =self.load_alignments(idx)
        assert frames.shape[0]==75, f"75 아니고 {frames.shape[0]}"
        assert frames.shape[1]==1, f"1 아니고 {frames.shape[1]}"
        assert alignments.shape[0]==30, f"50 아니고 {alignments.shape[0]}"
        return frames,alignments
    
    def load_video(self, idx):
        cap = cv2.VideoCapture(self.video_list[idx])
        frames = []
        transform = transforms.Grayscale()  # RGB to Grayscale 변환

        # 비디오의 모든 프레임을 읽어서 처리
        for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV에서는 BGR로 이미지를 로드하므로 RGB로 변환
            frame = torch.tensor(frame).permute(2, 0, 1)  # HWC to CHW
            frame = transform(frame.float())  # Grayscale 변환
            
            frame = frame[:, 190:236, 80:220]  # 프레임 자르기
            frames.append(frame)

        cap.release()  # 모든 프레임을 처리한 후, 비디오 캡처를 해제

        if frames:  # 프레임이 하나라도 존재하는 경우에만 처리 
            frames = torch.stack(frames)
            if frames.shape[0] < 75:
                padding = torch.zeros(75 - frames.shape[0],frames.shape[1],frames.shape[2],frames.shape[3])
                
                frames=torch.cat((frames, padding), dim=0)
            mean = torch.mean(frames.float(), dim=(0, 2, 3), keepdim=True)
            std = torch.std(frames.float(), dim=(0, 2, 3), keepdim=True)
            return (frames - mean) / std
        
    def load_alignments(self,idx):
        tokens = []
        with open(self.align_list[idx], 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip().split()
            if line and line[2] != 'sil':
                tokens.extend([' '] + list(line[2]))  # 공백을 추가하고 문자 단위로 분할하여 리스트에 추가

        # 문자열을 숫자로 변환
        num_tokens = char_to_num(tokens[1:])  # 첫 번째 공백을 제외하고 변환
        target_length = 30
        current_length = num_tokens.size(0)
        if current_length < target_length:
            # (앞쪽 패딩, 뒤쪽 패딩) 순서로 패딩을 추가
            pad_size = (0, target_length - current_length)  # 뒤쪽에만 패딩을 추가
            num_tokens = F.pad(num_tokens.view(1, -1), pad_size, "constant", 39).view(-1)
        elif current_length > target_length:
            num_tokens = num_tokens[:target_length]
        return num_tokens

class SimpleLipnet(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.conv_layer=nn.Sequential(
            nn.Conv3d(in_channels=75, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(in_channels=256, out_channels=75, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
        )
        self.lstm_layer=nn.ModuleList([
            nn.LSTM(input_size=85, 
                            hidden_size=128, 
                            bidirectional=True, 
                            batch_first=True),
            nn.Dropout(p=0.5),
            nn.LSTM(input_size=128*2, 
                            hidden_size=128, 
                            bidirectional=True, 
                            batch_first=True),
            nn.Dropout(p=0.5)]
        )
        self.linear=nn.Linear(128*2,vocab_size+1)
        init.kaiming_normal_(self.linear.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x=self.conv_layer(x)
        batch_size, seq_len, channels, height, width = x.size()
        x=x.view(batch_size, seq_len, -1)
        
        for layer in self.lstm_layer:
            if isinstance(layer, nn.LSTM):
                x, _ = layer(x)
            else:
                x = layer(x)
        x=F.log_softmax(self.linear(x), dim=-1)
        
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
        return x
    
def char_to_num(characters):
    return torch.tensor([char_to_index[char] for char in characters])

def nuxwm_to_char(numbers):
    return ''.join([index_to_char[number] for number in numbers.tolist()])

def compute_ctc_loss(y_pred, y_true):
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    batch_size=y_pred.shape[1]
    input_length = torch.full((batch_size,), y_pred.size(0), dtype=torch.long)  # [T, N, C]에서 T는 시퀀스 길이
    target_length = torch.full((batch_size,), y_true.size(1), dtype=torch.long)  # y_true의 형태는 [N, S] 입니다.
    
    loss = ctc_loss(y_pred, y_true, input_length, target_length)
    return loss

def produce_sample(decoder,model, data):
    model.eval()
    predictions = model(data)
    predictions = predictions.permute(1, 0, 2) #(T, N, C)-> (N, T, C)
    beam_results, beam_scores, timesteps, out_lens = decoder.decode(predictions)
    # print(beam_results[0][0][:out_lens[0][0]])
    return beam_results, beam_scores, timesteps, out_lens

def save_model(model,save_path):
    torch.save(model.state_dict(), save_path)
    print(save_path," 저장하였습니다.")
    
def load(model, accelerator,model_path, device):
        model_state_dict = torch.load(model_path, map_location=device)
        model = accelerator.unwrap_model(model)
        model.load_state_dict(model_state_dict)
        
    
def save(accelerator, model,model_path, optim_path, scheduler_path=None):
        model_state_dict = accelerator.get_state_dict(model)
        torch.save(model_state_dict, model_path)
        print(model_path," 저장하였습니다.")
       
    
def load_model(model,load_path):
    
    state_dict = torch.load(load_path)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict)
    return model
    
if __name__=="__main__":
    string_="abcdefghijklmnopqrstuvwxyz'?!123456789 "
    vocab = [x for x in string_]
    vocab_size=len(vocab)
    char_to_index = {char: index for index, char in enumerate(vocab)}
    index_to_char = {index: char for index, char in enumerate(vocab)}
    
    dataset=CustomDataset()
    total_size=len(dataset)
    train_size = int(total_size * 0.9)
    test_size = total_size-train_size
    print("train_size",train_size,"test_size",test_size)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, num_workers=2, batch_size=64, shuffle=True,drop_last=True)
    test_batch_size=4
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True,drop_last=True)
    
    kwargs_handler = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_process_kwargs = InitProcessGroupKwargs(timeout = timedelta(seconds = 1800))
    accelerator = Accelerator(
                kwargs_handlers = [DEFAULT_DDP_KWARGS, init_process_kwargs],
                log_with="wandb",gradient_accumulation_steps=2
    )    
    # accelerator = Accelerator(gradient_accumulation_steps=8)
    # accelerator = Accelerator(log_with="wandb",gradient_accumulation_steps=2,project_dir="/exp_logs/simple-lipnet")
    # accelerator.init_trackers(
    #     project_name="simple_lipnet", 
    #     config={"dropout": 0.1, "learning_rate": 1e-2},
    # )
    device=accelerator.device
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"Using GPU indices: {accelerator.state.process_index}")
    
    # model=SimpleLipnet(vocab_size).to(device)
    model=SimpleLipnet(vocab_size)
    # model.to(device)  # 모델을 적절한 device로 이동
    optimizer = Adam(model.parameters())
    scheduler=None
    train_loader, test_loader, model, optimizer = accelerator.prepare(
        train_loader, test_loader, model, optimizer
    )
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)
    accelerator.wait_for_everyone()
    test_loader_iter = iter(test_loader)
    accelerator.init_trackers(project_name="simple_lipnet")
    decoder = CTCBeamDecoder(
        labels="abcdefghijklmnopqrstuvwxyz'?!123456789 #",
        model_path=None,
        alpha=0,
        beta=0,
        cutoff_top_n=40,
        cutoff_prob=1.0,
        beam_width=100,
        num_processes=4,
        blank_id=0,
        log_probs_input=True
    )
    # load_path="/exp_logs/simple-lipnet/s1_9.pth"
    # load(model,accelerator,load_path,device)
    
    num_epochs=100
    for epoch in range(0,num_epochs):
        model.train()
        total_loss=0
        num=0
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            with accelerator.accumulate(model):
            # data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                predictions = model(data)
                loss = compute_ctc_loss(predictions, targets)*100
                num+=1
                total_loss=loss.item()
                accelerator.backward(loss)
                optimizer.step()
        accelerator.log({"training_loss": total_loss/len(train_loader)})
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader)}')
        
        
        try:
            data, target = next(test_loader_iter)
        except StopIteration:
            test_loader_iter = iter(test_loader)
            data, target = next(test_loader_iter)
        labels="abcdefghijklmnopqrstuvwxyz'?!123456789 #"
        beam_results, beam_scores, timesteps, out_lens=produce_sample(decoder,model, data)
        for i in range(test_batch_size):
            decoded = "".join(labels[n] for n in beam_results[i][0][:out_lens[i][0]])
            gt_decoded = "".join([labels[x] for x in target[i]])
            print("prediction : ",decoded,"\tground truth : ", gt_decoded)
        
    
        if (epoch + 1) % 3 == 0:
            save_path = f'/exp_logs/simple-lipnet/s1_{epoch+1}.pth'
            save(accelerator,model,save_path,None,None)
            
            
    accelerator.end_training()
    
