import copy
import logging
from sys import stderr

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm

from utils import debug
from modules.model import DevignModel
from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, NLLLoss

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from loss.utils import get_one_hot
from loss.loss_base_f import CrossEntropy



class FocalLoss(CrossEntropy):
    r"""
    Reference:
    Li et al., Focal Loss for Dense Object Detection. ICCV 2017.

        Equation: Loss(x, class) = - (1-sigmoid(p^t))^gamma \log(p^t)

    Focal loss tries to make neural networks to pay more attentions on difficult samples.

    Args:
        gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                               putting more focus on hard, misclassiﬁed examples
    """
    def __init__(self, para_dict=None):
        super(FocalLoss, self).__init__(para_dict)
        self.gamma = 2.0 #hyper-parameter
        self.sigmoid = nn.Sigmoid()
        self.num_class_list = [917, 1140, 469, 383, 205, 275, 249, 134, 164, 110, 138, 109, 80, 95, 75, 49, 32, 34, 31, 26, 26, 24, 20, 18, 12, 16, 10, 12, 25, 13, 133]
        self.weight_list = torch.FloatTensor(np.array([1 for _ in self.num_class_list])).to(self.device)
        self.num_classes = 31
    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        weight = (self.weight_list[targets]).to(targets.device) \
            if self.weight_list is not None else \
            torch.FloatTensor(torch.ones(targets.shape[0])).to(targets.device)
        label = get_one_hot(targets, self.num_classes)
        p = self.sigmoid(inputs)
        focal_weights = torch.pow((1-p)*label + p * (1-label), self.gamma)
        loss = F.binary_cross_entropy_with_logits(inputs, label, reduction = 'none') * focal_weights
        loss = (loss * weight.view(-1, 1)).sum() / inputs.shape[0]
        return loss


class ClassBalanceCE(CrossEntropy):
    r"""
    Reference:
    Cui et al., Class-Balanced Loss Based on Effective Number of Samples. CVPR 2019.

        Equation: Loss(x, c) = \frac{1-\beta}{1-\beta^{n_c}} * CrossEntropy(x, c)

    Class-balanced loss considers the real volumes, named effective numbers, of each class, \
    rather than nominal numeber of images provided by original datasets.

    Args:
        beta(float, double) : hyper-parameter for class balanced loss to control the cost-sensitive weights.
    """
    def __init__(self, para_dict= None):
        super(ClassBalanceCE, self).__init__()
        self.beta = 0.9999
        self.num_classes = 31
        self.num_class_list = [917, 1140, 469, 383, 205, 275, 249, 134, 164, 110, 138, 109, 80, 95, 75, 49, 32, 34, 31, 26, 26, 24, 20, 18, 12, 16, 10, 12, 25, 13, 133]
        self.class_balanced_weight = np.array([(1-self.beta)/(1- self.beta ** N) for N in self.num_class_list])
        self.class_balanced_weight = torch.FloatTensor(self.class_balanced_weight / np.sum(self.class_balanced_weight) * self.num_classes).to(self.device)
        self.weight_list = self.class_balanced_weight
        
    def update(self, epoch):
        """
        Args:
            epoch: int. starting from 1.
        """
        if False:
            self.weight_list = self.class_balanced_weight
        else:
            start = (epoch-1) // self.drw_start_epoch
            if start:
                self.weight_list = self.class_balanced_weight
                
class ClassBalanceFocal(CrossEntropy):
    r"""
    Reference:
    Li et al., Focal Loss for Dense Object Detection. ICCV 2017.
    Cui et al., Class-Balanced Loss Based on Effective Number of Samples. CVPR 2019.

        Equation: Loss(x, class) = \frac{1-\beta}{1-\beta^{n_c}} * FocalLoss(x, c)

    Class-balanced loss considers the real volumes, named effective numbers, of each class, \
    rather than nominal numeber of images provided by original datasets.

    Args:
        gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                               putting more focus on hard, misclassiﬁed examples
        beta(float, double): hyper-parameter for class balanced loss to control the cost-sensitive weights.
    """
    def __init__(self, para_dict=None):
        super(ClassBalanceFocal, self).__init__()
        self.beta = 0.999
        self.gamma = 0.5
        self.num_classes = 31
        self.num_class_list = [917, 1140, 469, 383, 205, 275, 249, 134, 164, 110, 138, 109, 80, 95, 75, 49, 32, 34, 31, 26, 26, 24, 20, 18, 12, 16, 10, 12, 25, 13, 133]
        
        self.class_balanced_weight = np.array([(1-self.beta)/(1- self.beta ** N) for N in self.num_class_list])
        self.class_balanced_weight = torch.FloatTensor(self.class_balanced_weight / np.sum(self.class_balanced_weight) * self.num_classes).to(self.device)
        self.sigmoid = nn.Sigmoid()
        self.weight_list = self.class_balanced_weight

    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        weight = (self.weight_list[targets]).to(targets.device)
        label = get_one_hot(targets, self.num_classes)
        p = self.sigmoid(inputs)
        focal_weights = torch.pow((1-p)*label + p * (1-label), self.gamma)
        loss = F.binary_cross_entropy_with_logits(inputs, label, reduction = 'none') * focal_weights
        loss = (loss * weight.view(-1, 1)).sum() / inputs.shape[0]
        return loss


def evaluate_metrics(model, loss_function, num_batches, data_iter):
    model.eval()
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        loss_function1 = CrossEntropyLoss()
        for _ in range(num_batches):
            graph, targets, seq = data_iter()
            targets = targets.cuda()
            seq = seq.cuda()
            #predictions = model(graph, seq, cuda=True)
            #batch_loss = loss_function(predictions, targets.long())
            predictions, seq_pre, graph_pre = model(graph, seq, cuda=True)

            #print(predictions)
            batch_loss = loss_function(seq_pre, targets.long()) + loss_function1(graph_pre, targets.long())
            #batch_loss = loss_function(seq_pre, targets.long()) 
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            if predictions.ndim == 2:
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
            all_targets.extend(targets.detach().cpu().numpy().tolist())
        model.train()
        return np.mean(_loss).item(), \
               accuracy_score(all_targets, all_predictions) * 100, \
               precision_score(all_targets, all_predictions, average='macro') * 100, \
               recall_score(all_targets, all_predictions, average='macro') * 100, \
               f1_score(all_targets, all_predictions, average='macro') * 100
    pass


def train(model, dataset, epoches, dev_every, loss_function, optimizer, save_path, log_every=5, max_patience=5):
    debug('Start Training')
    debug(dev_every)
    logging.info('Start training!')
    train_losses = []
    best_model = None
    patience_counter = 0
    best_acc = 0
    log_flag = 0
    max_steps = epoches * dev_every
    all_train_acc = []
    all_train_loss = []
    all_valid_acc = []
    all_valid_loss = []
    
    loss_function1 = ClassBalanceFocal()#FocalLoss()
    #loss_function1 = CrossEntropyLoss()
    loss_function1.cuda()
    try:
        for step_count in range(max_steps):
            #print("begin training")
            #print(step_count % dev_every)
            #if(step_count % dev_every==0):
            #    continue
            #train
            model.train()
            #gradient = 0
            model.zero_grad()
            graph, targets, seq = dataset.get_next_train_batch()   #first
            #print(dataset)
            #print(targets.shape)
            targets = targets.cuda()
            #if len(seq) < 512 * 128:
            #    k = 
            #print(seq.shape)
            seq = seq.cuda()
            predictions, head_pre, tail_pre = model(graph, seq, cuda=True)

            #print(predictions)
            epoch_now = int(step_count / dev_every)
            epoch_norm = float(epoch_now) / epoches
            alpha = epoch_norm * epoch_norm
            
            
            
            batch_loss = (alpha) * loss_function1(head_pre, targets.long()) + (1- alpha) * loss_function(tail_pre, targets.long())
            
            '''
            if log_every is not None and (step_count % log_every == log_every - 1):
                debug('Step %d\t\tTrain Loss %10.3f' % (step_count, batch_loss.detach().cpu().item()))
                logging.info('Step %d\t\tTrain Loss %10.3f' % (step_count, batch_loss.detach().cpu().item()))
            '''
            #print(batch_loss.detach().cpu().item())
            #train_losses.append(batch_loss.detach().cpu().item())
            train_losses.append(batch_loss.detach().item())
            batch_loss.backward()
            optimizer.step()
            
                #print(DevignModel.state_dict(model))
            
            if step_count % dev_every == (dev_every - 1):
                #print(step_count % dev_every)
                log_flag += 1
                debug('@@@' * 35)
                

                debug(step_count)
                debug(log_flag)
                train_loss, train_acc, train_pr, train_rc, train_f1 = evaluate_metrics(model, loss_function, dataset.initialize_train_batch(), dataset.get_next_train_batch)
                all_train_acc.append(train_acc)
                all_train_loss.append(train_loss)

                logging.info('-' * 100)
                logging.info('Epoch %d\t---Train--- Average Loss: %10.4f\t Patience %d\t Loss: %10.4f\tAccuracy: %0.4f\tPrecision: %0.4f\tRecall: %0.4f\tf1: %5.3f\t' % (
                    log_flag, np.mean(train_losses).item(), patience_counter, train_loss, train_acc, train_pr, train_rc, train_f1))
                loss, acc, pr, rc, valid_f1 = evaluate_metrics(model, loss_function, dataset.initialize_valid_batch(), dataset.get_next_valid_batch)
                logging.info('Epoch %d\t----Valid---- Loss: %0.4f\tAccuracy: %0.4f\tPrecision: %0.4f\tRecall: %0.4f\tF1: %0.4f' % (log_flag, loss, acc, pr, rc, valid_f1))
                
                test_loss, test_acc, test_pr, test_rc, test_f1 = evaluate_metrics(model, loss_function, dataset.initialize_test_batch(), dataset.get_next_test_batch)
                logging.info('Epoch %d\t----Test---- Loss: %0.4f\tAccuracy: %0.4f\tPrecision: %0.4f\tRecall: %0.4f\tF1: %0.4f' % (log_flag, test_loss, test_acc, test_pr, test_rc, test_f1))
                all_valid_acc.append(acc)
                all_valid_loss.append(loss)
                if acc > best_acc:
                    patience_counter = 0
                    best_acc = acc
                    best_model = copy.deepcopy(model.state_dict())
                    _save_file = open(save_path + str(log_flag) + '-model.bin', 'wb')
                    torch.save(model.state_dict(), _save_file)
                    _save_file.close()
                else:
                    _save_file = open(save_path + str(log_flag) + '-model.bin', 'wb')
                    torch.save(model.state_dict(), _save_file)
                    _save_file.close()
                    patience_counter += 1
                train_losses = []
                if patience_counter == max_patience:
                    break
    except KeyboardInterrupt:
        debug('Training Interrupted by user!')
        logging.info('Training Interrupted by user!')
    logging.info('Finish training!')

    if best_model is not None:
        model.load_state_dict(best_model)
    _save_file = open(save_path + '-model.bin', 'wb')
    torch.save(model.state_dict(), _save_file)
    _save_file.close()


    #model.load_state_dict(torch.load('./Models/FFmpeg/'+'DevignModel187-model.bin'))
    #torch.no_grad()
    logging.info('#' * 100)
    logging.info("Test result")
    loss, acc, pr, rc, f1 = evaluate_metrics(model, loss_function, dataset.initialize_test_batch(),
                                       dataset.get_next_test_batch)
    debug('%s\tTest Accuracy: %0.2f\tPrecision: %0.2f\tRecall: %0.2f\tF1: %0.2f' % (save_path, acc, pr, rc, f1))
    logging.info('%s\t----Test---- Loss: %0.4f\tAccuracy: %0.4f\tPrecision: %0.4f\tRecall: %0.4f\tF1: %0.4f' % (save_path, loss, acc, pr, rc, f1))
    

    import os
    #5、需要修改路径
    if not os.path.exists('models/Fan_dataset/'):
        os.makedirs('models/Fan_dataset/')
    with open('models/Fan_dataset/train_acc.txt', 'w', encoding='utf-8') as f:
        for i in all_train_acc:
            f.writelines(str(i) + '\n')
    with open('models/Fan_dataset/train_loss.txt', 'w', encoding='utf-8') as f:
        for i in all_train_loss:
            f.writelines(str(i) + '\n')
    with open('models/Fan_dataset/valid_acc.txt', 'w', encoding='utf-8') as f:
        for i in all_valid_acc:
            f.writelines(str(i) + '\n')
    with open('models/Fan_dataset/valid_loss.txt', 'w', encoding='utf-8') as f:
        for i in all_valid_loss:
            f.writelines(str(i) + '\n')

