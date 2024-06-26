from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Font_Val, Dataset_Pred, Dataset_Font
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

import pickle as pkl

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, get_acc_p_r_f1, get_acc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from torch.autograd import Variable

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')

def loss_2_type(targets, logits, logits_cls, cls_label, mask): # label, pred, pred_cls, mask
    targets_cls = cls_label #(targets* mask)[:,:,0]
    #mask_cls = torch.cat((mask[:,:,:1], mask[:,:,:1]),2)
    #logits_cls = (logits_cls*mask_cls)#[:,:,0]
    loss_cls = F.cross_entropy(logits_cls.reshape(-1, logits_cls.size(-1)), torch.squeeze(targets_cls.long().view(-1,1)), ignore_index=-1)
    loss_reg = F.mse_loss((logits*mask), (targets* mask))
    loss = loss_cls + loss_reg
    return loss

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
        # self.train_data, self.train_loader = self._get_data(flag = 'train_font')
        # self.vali_data, self.vali_loader = self._get_data(flag = 'val_font')
        #self.test_data, self.test_loader = self._get_data(flag = 'test_font')

        self.CMD_ARGS_MASK = torch.tensor([[ 1, 1, 0, 0, 0, 0, 1, 1],   # m
                                  [ 1, 1, 0, 0, 0, 0, 1, 1],   # l
                                  [ 1, 1, 1, 1, 1, 1, 1, 1],   # c
                                  [ 1, 1, 0, 0, 0, 0, 1, 1],   # a
                                  [ 0, 0, 0, 0, 0, 0, 0, 0],   # EOS
                                  [ 0, 0, 0, 0, 0, 0, 0, 0],   # SOS
                                  [ 0, 0, 0, 0, 0, 0, 0, 0]])  # z

    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
            'Fonts': Dataset_Font,
            'Fonts_Val': Dataset_Font_Val,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1
        #root_path = args.root_path
        data_set = []
        if flag == 'test_font_with_name':
            Data = data_dict['Fonts_Val']
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
            data_set = Data(
            root_path=args.val_font,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )

        elif flag == 'test_font' or flag == 'val_font':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
            data_set = Data(
            root_path=args.val_font,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        elif flag == 'train_font':
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
            data_set = Data(
            root_path=args.train_font,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        elif flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
            data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
            root_path = args.root_path
            data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
            root_path = args.root_path
            data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
            
        
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,cls_label,mask, name,label) in enumerate(vali_loader):
            #print(i)
            cls_label = cls_label.float().to(self.device)
            label = torch.tensor(label).long().to(self.device)
            mask = mask.float().to(self.device)
            pred = self._process_one_batch(
                    vali_data, batch_x, batch_y, cls_label, mask, )

            loss = F.cross_entropy(pred.reshape(-1, pred.size(-1)), torch.squeeze(label.long().view(-1,1)), ignore_index=-1)

            #loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss.detach().cpu().numpy())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        self.train_data, self.train_loader = self._get_data(flag = 'train_font')
        train_data = self.train_data
        train_loader = self.train_loader
        #= self._get_data(flag = 'train_font')
        self.vali_data, self.vali_loader = self._get_data(flag = 'val_font')
        vali_data = self.vali_data
        vali_loader = self.vali_loader
        
        test_data = self.vali_data
        test_loader = self.vali_loader

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,cls_label,mask, name,label) in enumerate(train_loader):
                iter_count += 1
                cls_label = cls_label.to(self.device)
                mask = mask.float().to(self.device)
                label = torch.tensor(label).long().to(self.device)
                
                model_optim.zero_grad()
                pred = self._process_one_batch(
                    train_data, batch_x, batch_y, cls_label, mask )

                loss = F.cross_entropy(pred.reshape(-1, pred.size(-1)), torch.squeeze(label.long().view(-1,1)), ignore_index=-1)

                #loss = loss_2_type(true[:,1:,:], pred, pred_cls, cls_label[:,1:], mask[:,1:,:])#criterion(pred, true)
                
                # probs = F.softmax(pred_cls, dim=-1)
                # _, pred_class = torch.topk(probs, k=1, dim=-1)
                # pred_class = 0

                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            model_path = path+'/'+'checkpoint_'+str(epoch)+'_'+str(test_loss)+'.pth'
            torch.save(self.model.state_dict(), model_path)
            #self.model.load_state_dict(torch.load(model_path))
            early_stopping(vali_loss, self.model, path)

            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting, load, ii):
        # test_data = self.test_data
        # test_loader = self.test_loader

        test_data, test_loader = self._get_data(flag = 'test_font_with_name')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        
        self.model.eval()
        
        preds = None
        trues = None

        # result save
        folder_path = './results_deepsvg_format/cls_task/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        class_lable = []
        class_pred = []
        flag = False
        for i, (batch_x,batch_y,cls_label,mask,name, label) in enumerate(test_loader):
            cls_label = cls_label.to(self.device)
            mask = mask.float().to(self.device)
            pred = self._process_one_batch(
                    test_data, batch_x, batch_y, cls_label, mask )
            # pred, true, pred_cls, _, _ = self._process_one_batch(
            #     test_data, batch_x, batch_y, cls_label, mask)
            #loss = loss_2_type(true, pred, pred_cls,cls_label, mask)

            

            probs = F.softmax(pred, dim=-1)
            _, pred_class = torch.topk(probs, k=1, dim=-1)

            if not flag:
                preds = pred_class.detach().cpu().numpy().reshape(-1)
                trues = label.detach().cpu().numpy().reshape(-1)
                flag = True
            
            else:
                preds = np.concatenate((preds,pred_class.detach().cpu().numpy().reshape(-1)))
                trues = np.concatenate((trues,label.detach().cpu().numpy().reshape(-1)))
            # class_lable.append(label.detach().cpu().numpy().reshape(-1))
            # class_pred.append(pred_class.detach().cpu().numpy().reshape(-1))
        # trues = np.array(trues.cpu().numpy())
        # preds = np.array(preds.cpu().numpy())
        precision, recall, f1 = get_acc_p_r_f1(trues,preds )
        print("precision, recall, f1:", precision, recall, f1)
        print(get_acc(trues,preds))
        print(classification_report(trues,preds))

        return

    def test_with_rev(self, setting, load, ii):
        # test_data = self.test_data
        # test_loader = self.test_loader

        test_data, test_loader = self._get_data(flag = 'test_font_with_name')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        
        self.model.eval()
        
        preds = None
        trues = None

        # result save
        folder_path = './results_deepsvg_format/test_with_rev_input_mask/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        results_pred = {}
        results_lable = {}
        results_enc_out = {} 
        results_dec_inp = {}
        results_input = {}

        
        for i, (batch_x,batch_y,cls_label,mask,name) in enumerate(test_loader):
            cls_label = cls_label.to(self.device)
            mask = mask.float().to(self.device)
            pred, true, pred_cls, enc_out, dec_inp = self._process_one_batch(
                test_data, batch_x, batch_y, cls_label, mask)
            #loss = loss_2_type(true, pred, pred_cls,cls_label, mask)

            

            probs = F.softmax(pred_cls, dim=-1)
            _, pred_class = torch.topk(probs, k=1, dim=-1)
            data_pad = -1*torch.ones((true.shape[0],true.shape[1],5)).to(self.device)
            #pred[:,:,0] = pred_class[:,:,0]
            
            pred_class = torch.cat((cls_label.unsqueeze(-1)[:,:1,:],pred_class),1)

            pred_mask = self.CMD_ARGS_MASK[pred_class.long()].squeeze().to(self.device)

            pred = torch.cat((true[:,:1,:],pred),1)
            #pred = pred*mask
            pred = pred*pred_mask
            pred_tmp = pred.reshape(-1,8).detach().cpu().numpy()
            pred_tmp[pred_tmp.sum(axis=1)==0,:] = -1
            pred = torch.tensor(pred_tmp.reshape(pred.shape[0],pred.shape[1],-1)).to(self.device)
            
            #mask by pred
            
            cls_label = cls_label.reshape(-1,1).detach().cpu().numpy()
            pred_class = pred_class.reshape(-1,1).detach().cpu().numpy()
            pred_class[cls_label==-1] = -1
            pred_class = torch.tensor(pred_class.reshape(pred.shape[0],pred.shape[1],-1)).to(self.device)
            cls_label = torch.tensor(cls_label.reshape(pred.shape[0],pred.shape[1],-1)).to(self.device)

            
            
            pred = torch.cat((pred_class,data_pad,pred),2)
            # layouts = torch.cat((x[:, :1], pred[:, :, 0]), dim=1).detach().cpu().numpy()
             #(pred*mask).detach().cpu().numpy()
            true = torch.cat((cls_label,data_pad,true),2)
            # np.save('./layouts_pred.csv', layouts)
            # np.save('./layouts_label.csv', x.cpu().numpy())
            
            p_data = pred.detach().cpu().numpy()
            #true = true.detach().cpu().numpy()
            enc_out = enc_out.detach().cpu().numpy()
            dec_inp = dec_inp.detach().cpu().numpy()
            #batch_x = batch_x.detach().cpu().numpy()
            for i in range(len(p_data)):
                results_pred[name[i]] = p_data[i]
                results_lable[name[i]] = true[i]
                results_enc_out[name[i]] = enc_out[i]
                results_dec_inp[name[i]] = dec_inp[i]
                results_input[name[i]] = batch_x[i] # for inter task's mask matrix
                #np.save(folder_path+ name[i] +'.npy', p_data[i])

            if preds is not None:
                preds = np.concatenate((preds,pred.detach().cpu().numpy()),0)
                trues = np.concatenate((trues,true.detach().cpu().numpy()),0)
            else:
                preds = pred.detach().cpu().numpy()
                trues = true.detach().cpu().numpy()


        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        #print('test shape:', preds.shape, trues.shape)

        

        # mae, mse, rmse, mape, mspe = metric(preds, trues)
        # print('mse:{}, mae:{}'.format(mse, mae))

        # np.save(folder_path+'metrics_' +str(ii) +'.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred_' +str(ii) +'.npy', preds)
        np.save(folder_path+'true_' +str(ii) +'.npy', trues)
        file = open(folder_path+'pred_by_name.pkl','wb')
        pkl.dump(results_pred, file)
        file.close()
        file = open(folder_path+'true_by_name.pkl','wb')
        pkl.dump(results_lable, file)
        file.close()
        file = open(folder_path+'enc_by_name.pkl','wb')
        pkl.dump(results_enc_out, file)
        file.close()
        file = open(folder_path+'dec_by_name.pkl','wb')
        pkl.dump(results_dec_inp, file)
        file.close()
        file = open(folder_path+'input_by_name.pkl','wb')
        pkl.dump(results_input, file)
        file.close()

        return

    def retrieval(self, setting, load, ii):
        # test_data = self.test_data
        # test_loader = self.test_loader

        test_data, test_loader = self._get_data(flag = 'test_font_with_name')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        
        self.model.eval()
        
        preds = None
        trues = None

        # result save
        folder_path = './results_deepsvg_format/retrieval_task/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        results_pred = {}
        results_lable = {}
        results_dec = {}

        
        for i, (batch_x,batch_y,cls_label,mask,name) in enumerate(test_loader):
            cls_label = cls_label.to(self.device)
            mask = mask.float().to(self.device)
            hidden_enc, batch_y, input_dec = self._process_one_batch_retrieval(
                test_data, batch_x, batch_y, cls_label, mask) # 
            #loss = loss_2_type(true, pred, pred_cls,cls_label, mask)

            batch_x = batch_x.float().to(self.device)
            #batch_y = batch_y.float()

            p_data = hidden_enc.detach().cpu().numpy()
            input_dec = input_dec.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()
            for i in range(len(p_data)):
                results_pred[name[i]] = p_data[i]
                results_lable[name[i]] = batch_y[i]
                results_dec[name[i]] = input_dec[i]
        file = open(folder_path+'enc_by_name.pkl','wb')
        pkl.dump(results_pred, file)
        file.close()
        file = open(folder_path+'input_by_name.pkl','wb')
        pkl.dump(results_lable, file)
        file.close()
        file = open(folder_path+'dec_by_name.pkl','wb')
        pkl.dump(results_dec, file)
        file.close()

        
        return
    
    def encode(self, setting, x_input, svg=[[0]], load = True):
        # test_data = self.test_data
        # test_loader = self.test_loader
        cls_label = svg #torch.tensor(svg[:,:1]).unsqueeze(0).long().to(self.device)
        
        x_input = torch.tensor(x_input).float().unsqueeze(0).to(self.device)
        
        mask = (x_input !=-1).float().to(self.device)

        #test_data, test_loader = self._get_data(flag = 'test_font_with_name')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        
        self.model.eval()
        
        enc_out = self._process_one_batch_encoder(
                x_input, cls_label, mask)

        return enc_out
        
        
        

    def decode(self, setting, enc_out, x_dec, x_input, svg=[[0]], load = True):
        # test_data = self.test_data
        # test_loader = self.test_loader
        cls_label = torch.tensor(svg[:,:1]).unsqueeze(0).long().to(self.device)
        
        x_input = torch.tensor(x_input).float().unsqueeze(0).to(self.device)
        
        mask = (x_input !=-1).float().to(self.device)

        #test_data, test_loader = self._get_data(flag = 'test_font_with_name')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        
        self.model.eval()
        
        pred, pred_cls, enc_out = self._process_one_batch_interpolation_with_enc(
               enc_out, x_dec, x_input, x_input, cls_label, mask)

        #return enc_out.detach().cpu().numpy()[0]
        
        
        #'''
        probs = F.softmax(pred_cls, dim=-1)
        _, pred_class = torch.topk(probs, k=1, dim=-1)
        data_pad = -1*torch.ones((x_input.shape[0],x_input.shape[1],5)).to(self.device)
            #pred[:,:,0] = pred_class[:,:,0]
            
        pred_class = torch.cat((cls_label[:,:1,:],pred_class),1)[:x_input.shape[0]]

        pred_mask = self.CMD_ARGS_MASK[pred_class.long()].squeeze().to(self.device)
        
        

        pred = torch.cat((x_input[:,:1,:],pred),1)[:x_input.shape[0],:,:]
        #pred = pred*mask
        pred = pred*pred_mask
        pred_tmp = pred.reshape(-1,8).detach().cpu().numpy()
        pred_tmp[pred_tmp.sum(axis=1)==0,:] = -1
        pred = torch.tensor(pred_tmp.reshape(pred.shape[0],pred.shape[1],-1)).to(self.device)
            
            
            
        
        #pred_class = pred_class.reshape(-1,1).detach().cpu().numpy()
        #cls_label = cls_label.reshape(-1,1).detach().cpu().numpy()
        pred_class[cls_label==-1] = -1
        #pred_class = torch.tensor(pred_class.reshape(pred.shape[0],pred.shape[1],-1)).float().to(self.device)
        
            
            
        pred = torch.cat((pred_class.float(),data_pad,pred),2)
        #pred = torch.cat((cls_label.float(),data_pad,pred),2)
        
           
        return pred, enc_out
        #'''
    

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = None
        trues = None
        
        for i, (batch_x,batch_y,cls_label,mask) in enumerate(pred_loader):
            cls_label = cls_label.float().to(self.device)
            mask = mask.float().to(self.device)
            pred, true, pred_cls = self._process_one_batch(
                pred_data, batch_x, batch_y, cls_label, mask)
            # loss = loss_2_type(true, pred, pred_cls, mask)

            # pred, true, pred_cls = self._process_one_batch(
            #     pred_data, batch_x, batch_y, cls_label, mask)
            probs = F.softmax(pred_cls, dim=-1)
            _, pred_class = torch.topk(probs, k=1, dim=-1)
            pred[:,:,0] = pred_class[:,:,0]
            # layouts = torch.cat((x[:, :1], pred[:, :, 0]), dim=1).detach().cpu().numpy()
            layouts = (pred*mask).detach().cpu().numpy()
            true[:,:,0] = true[:,:,0]-1
            true = true * mask
            # np.save('./layouts_pred.csv', layouts)
            # np.save('./layouts_label.csv', x.cpu().numpy())
            if preds is not None:
                preds = np.concatenate((preds,pred.detach().cpu().numpy()),0)
                trues = np.concatenate((trues,true.detach().cpu().numpy()),0)
            else:
                preds = pred.detach().cpu().numpy()
                trues = true.detach().cpu().numpy()
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, cls_label, mask):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        cls_label = cls_label.float().to(self.device)
        mask = mask.float().to(self.device)

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], batch_y.shape[1]-1, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], batch_y.shape[1]-1, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:1,:], dec_inp], dim=1).float().to(self.device) # 32, 48, 7 -> 32, 72, 7
        
        outputs = self.model(batch_x, cls_label, dec_inp, mask)
        # if self.args.inverse:
        #     outputs = dataset_object.inverse_transform(outputs) # here we need the ori dataset object -- by defu
        # f_dim = -1 if self.args.features=='MS' else 0
        # #batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        # batch_y = batch_y.to(self.device)


        return outputs #, batch_y, outputs_cls, enc_out, dec_inp

    def _process_one_batch_retrieval(self, dataset_object, batch_x, batch_y, cls_label, mask, retrival = True, reduce_hid = False):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        cls_label = cls_label.float().to(self.device)
        mask = mask.float().to(self.device)

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], batch_y.shape[1]-1, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], batch_y.shape[1]-1, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:1,:], dec_inp], dim=1).float().to(self.device) # 32, 48, 7 -> 32, 72, 7
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    enc_input = self.model(batch_x, cls_label, dec_inp, mask, retrival = retrival, reduce_hid = reduce_hid)[0]
                else:
                    enc_input = self.model(batch_x, cls_label, dec_inp, mask, retrival = retrival, reduce_hid = reduce_hid)
        else:
            if self.args.output_attention:
                enc_input, outputs_cls, _ = self.model(batch_x, cls_label, dec_inp, mask, retrival = retrival, reduce_hid = reduce_hid)
                #outputs_cls = self.model(batch_x, cls_label, dec_inp, mask)
            else:
                enc_input, outputs_cls = self.model(batch_x, cls_label, dec_inp, mask, retrival = retrival, reduce_hid = reduce_hid)
       
        #batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        batch_y = batch_y.to(self.device)

        return enc_input, batch_y, dec_inp
    
    def _process_one_batch_interpolation(self, enc_out, x_dec, batch_x, batch_y, cls_label, mask, retrival = True, reduce_hid = False):
        enc_out = torch.tensor(enc_out).float().unsqueeze(0).to(self.device)
        x_dec = torch.tensor(x_dec).float().unsqueeze(0).to(self.device)
        dec_inp = x_dec

        # decoder input
        # if self.args.padding==0:
        #     dec_inp = torch.zeros([batch_y.shape[0], batch_y.shape[1]-1, batch_y.shape[-1]]).float()
        # elif self.args.padding==1:
        #     dec_inp = torch.ones([batch_y.shape[0], batch_y.shape[1]-1, batch_y.shape[-1]]).float()
        # dec_inp = torch.cat([batch_y[:,:1,:], dec_inp], dim=1).float().to(self.device) # 32, 48, 7 -> 32, 72, 7
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model.decode_exp(enc_out, x_dec)[0]
                else:
                    outputs = self.model.decode_exp(enc_out, x_dec)
        else:
            if self.args.output_attention:
                outputs, outputs_cls, _ = self.model.decode_exp(enc_out, x_dec)
                #outputs_cls = self.model(batch_x, cls_label, dec_inp, mask)
            else:
                outputs, outputs_cls, enc_out = self.model.forward_single(enc_out, x_dec, batch_x)
        #batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        #batch_y = batch_y.to(self.device)


        return outputs, outputs_cls, enc_out #enc_input, batch_y, dec_inp

    def _process_one_batch_interpolation_with_enc(self, enc_out, x_dec, batch_x, batch_y, cls_label, mask, retrival = True, reduce_hid = False):
        # enc_out = torch.tensor(enc_out).float().unsqueeze(0).to(self.device)
        # enc_out=Variable(enc_out,requires_grad=True)
        x_dec = torch.tensor(x_dec).float().unsqueeze(0).to(self.device)
        dec_inp = x_dec

        # decoder input
        # if self.args.padding==0:
        #     dec_inp = torch.zeros([batch_y.shape[0], batch_y.shape[1]-1, batch_y.shape[-1]]).float()
        # elif self.args.padding==1:
        #     dec_inp = torch.ones([batch_y.shape[0], batch_y.shape[1]-1, batch_y.shape[-1]]).float()
        # dec_inp = torch.cat([batch_y[:,:1,:], dec_inp], dim=1).float().to(self.device) # 32, 48, 7 -> 32, 72, 7
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model.decode_exp(enc_out, x_dec)[0]
                else:
                    outputs = self.model.decode_exp(enc_out, x_dec)
        else:
            if self.args.output_attention:
                outputs, outputs_cls, _ = self.model.decode_exp(enc_out, x_dec)
                #outputs_cls = self.model(batch_x, cls_label, dec_inp, mask)
            else:
                outputs, outputs_cls, enc_out = self.model.forward_with_enc(enc_out, x_dec, batch_x)
        #batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        #batch_y = batch_y.to(self.device)


        return outputs, outputs_cls, enc_out #enc_input, batch_y, dec_inp

    def _process_one_batch_encoder(self, batch_x, cls_label, mask, retrival = True, reduce_hid = False):
        # enc_out = torch.tensor(enc_out).float().unsqueeze(0).to(self.device)
        # enc_out=Variable(enc_out,requires_grad=True)
        # x_dec = torch.tensor(x_dec).float().unsqueeze(0).to(self.device)
        # dec_inp = x_dec
        batch_y = batch_x

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], batch_y.shape[1]-1, batch_y.shape[-1]]).float().to(self.device)
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], batch_y.shape[1]-1, batch_y.shape[-1]]).float().to(self.device)
        dec_inp = torch.cat([batch_y[:,:1,:], dec_inp], dim=1).float().to(self.device) # 32, 48, 7 -> 32, 72, 7
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model.decode_exp(enc_out, x_dec)[0]
                else:
                    outputs = self.model.decode_exp(enc_out, x_dec)
        else:
            if self.args.output_attention:
                outputs, outputs_cls, _ = self.model.decode_exp(enc_out, x_dec)
                #outputs_cls = self.model(batch_x, cls_label, dec_inp, mask)
            else:
                enc_out, _ = self.model(batch_x, cls_label, dec_inp, mask, retrival = True)
        #batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        #batch_y = batch_y.to(self.device)


        return enc_out #enc_input, batch_y, dec_inp

