import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import tqdm
import numpy as np
from sklearn.metrics import r2_score

from dataSetLoad import Hyper_IMG
from retrieveBest import HybridSN

from myDefine import setup_seed
from myLoss import R2Loss
from myTest import TestAll
from myLog import log

import time



trainModel = True # True  False
num_epochs = 500

modelnameTrain = 'trainBest.pth'
modelnameVal = 'valBest.pth'
modelnameTest = 'testBest.pth'
filePath = r'../../'
logName = time.strftime('%Y-%m-%d_%H-%M-%S.txt', time.localtime(time.time()))

setup_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = Hyper_IMG(filePath,train="Train",transform=transforms.Compose([transforms.ToTensor()]))
val_dataset = Hyper_IMG(filePath,train="Val",transform=transforms.ToTensor())
test_dataset = Hyper_IMG(filePath,train="Test",transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

model = HybridSN(rate=16, class_num=2, windowSize=16, K=204)

criterion = nn.MSELoss()
criterion1 = R2Loss()

if trainModel:
    lossMin = 2
    R2Max1, R21Max1, R22Max1 = -200, -200, -200
    R2Max2, R21Max2, R22Max2 = -200, -200, -200
    for epoch in range(num_epochs):
        if epoch % 100 == 0:
            print('lr =', 0.001/(10**(epoch//100)))
            optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': 0.001/(10**(epoch//100))}], lr=0.001/(10**(epoch//100))) 
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100,
                                                                        eta_min=0, last_epoch=-1)
            
        model.train()
        model.to(device)
        lossTatol = 0
        t = tqdm.tqdm(enumerate(train_loader),desc = f'[train]')
        for step, (img, img_soilUp, img_soilDwon, img_stem, label) in t:
            output = model(img.permute(0, 1, 4, 2, 3).to(device), img_soilUp.permute(0, 1, 4, 2, 3).to(device), img_soilDwon.permute(0, 1, 4, 2, 3).to(device), img_stem.permute(0, 1, 4, 2, 3).to(device))
            
            regularization_loss = 0
            for param in model.parameters():
                regularization_loss += torch.sum(abs(param))
            loss = criterion1(output, label.to(device)) + criterion(output, label.to(device)) + 1e-7 * regularization_loss
            
            lossTatol += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
            
            
        lossAverage = lossTatol/(step+1)
        try:
            log('Epoch [{}/{}], AverageLoss: {:.4f}, loss: {:.4f}, lr: {}'.format(epoch+1, num_epochs, lossAverage, loss.item(), optimizer.state_dict()['param_groups'][0]['lr']), logName)
            torch.save(model.state_dict(),modelnameTrain)
        except:
            print('Model Save Error!!!')

        outputList = []
        labelList = []
        model.eval()

        # 模型测试
        lossTatol = 0
        t = tqdm.tqdm(enumerate(val_loader),desc = f'[Val]') # ,loss:{loss.item()}
        for step, (img, img_soilUp, img_soilDwon, img_stem, label) in t:
            output = model(img.permute(0, 1, 4, 2, 3).to(device), img_soilUp.permute(0, 1, 4, 2, 3).to(device), img_soilDwon.permute(0, 1, 4, 2, 3).to(device), img_stem.permute(0, 1, 4, 2, 3).to(device))
            loss = criterion(output, label.to(device)) + criterion1(output, label.to(device))
            
            lossTatol += loss.item()
            outputList.append(np.array(output[0].cpu().detach().numpy()))
            labelList.append(np.array(label[0].cpu().detach().numpy()))

        lossAverage = lossTatol/(step+1)
        outputList = np.array(outputList)
        labelList = np.array(labelList)

        R2 = r2_score(outputList, labelList) 
        R21 = r2_score(outputList[:, 0], labelList[:, 0])
        R22 = r2_score(outputList[:, 1], labelList[:, 1])
        if R2 > R2Max1:
            R2Max1 = R2
            R21Max1 = R21
            R22Max1 = R22
            try:
                torch.save(model.state_dict(),modelnameVal)
                log('Model Saved!! R2Max: {:.4f}, R21Max: {:.4f}, R22Max: {:.4f}'.format(R2Max1, R21Max1, R22Max1), logName)
            except:
                print('Model Save Error!!!')
        else:
            log('R2Max: {:.4f}, R21Max: {:.4f}, R22Max: {:.4f}'.format(R2Max1, R21Max1, R22Max1), logName)


        outputList = []
        labelList = []
        model.eval()

        # 模型测试
        lossTatol = 0
        t = tqdm.tqdm(enumerate(test_loader),desc = f'[Test]')
        for step, (img, img_soilUp, img_soilDwon, img_stem, label) in t:
            output = model(img.permute(0, 1, 4, 2, 3).to(device), img_soilUp.permute(0, 1, 4, 2, 3).to(device), img_soilDwon.permute(0, 1, 4, 2, 3).to(device), img_stem.permute(0, 1, 4, 2, 3).to(device))
            loss = criterion(output, label.to(device)) + criterion1(output, label.to(device))
            
            lossTatol += loss.item()
            outputList.append(np.array(output[0].cpu().detach().numpy()))
            labelList.append(np.array(label[0].cpu().detach().numpy()))

        lossAverage = lossTatol/(step+1)
        outputList = np.array(outputList)
        labelList = np.array(labelList)

        R2 = r2_score(outputList, labelList) 
        R21 = r2_score(outputList[:, 0], labelList[:, 0])
        R22 = r2_score(outputList[:, 1], labelList[:, 1])
        if R2 > R2Max2:
            R2Max2 = R2
            R21Max2 = R21
            R22Max2 = R22
            try:
                torch.save(model.state_dict(),modelnameTest)
                log('Model Saved!! R2Max: {:.4f}, R21Max: {:.4f}, R22Max: {:.4f}'.format(R2Max2, R21Max2, R22Max2), logName)
            except:
                print('Model Save Error!!!')
        else:
            log('R2Max: {:.4f}, R21Max: {:.4f}, R22Max: {:.4f}'.format(R2Max2, R21Max2, R22Max2), logName)
        print(' ')
        
TestAll(modelnameTrain, train_dataset, val_dataset, test_dataset, model, device, criterion, criterion1)
TestAll(modelnameVal, train_dataset, val_dataset, test_dataset, model, device, criterion, criterion1)
TestAll(modelnameTest, train_dataset, val_dataset, test_dataset, model, device, criterion, criterion1)


'''
[Test]: 1143it [00:03, 325.59it/s]
lossAverage: 0.09975876085427632 loss: 0.018110981211066246 MSE1: 0.0041492134 MAE1: 0.048976146 R21: 0.6545688216156549 MSE2: 0.0004266309 MAE2: 0.015676074 R22: 0.6527518798688585
[Test]: 143it [00:00, 308.85it/s]
lossAverage: 18.542519886978248 loss: 0.0311762485653162 MSE1: 0.005431875 MAE1: 0.04999218 R21: 0.5506259015330437 MSE2: 0.0003851711 MAE2: 0.014916257 R22: 0.6888732720596409
[Test]: 145it [00:00, 281.55it/s]
lossAverage: 0.1585997858047935 loss: 0.050016485154628754 MSE1: 0.0042366525 MAE1: 0.049788266 R21: 0.6781036425833643 MSE2: 0.0004365281 MAE2: 0.01599607 R22: 0.6740289343331869

[Test]: 1143it [00:03, 324.52it/s]
lossAverage: 0.11519754594266769 loss: 0.021116362884640694 MSE1: 0.004896068 MAE1: 0.052932855 R21: 0.631270000444443 MSE2: 0.00049608096 MAE2: 0.016871996 R22: 0.6269637851256782
[Test]: 143it [00:00, 283.73it/s]
lossAverage: 21.219560565298227 loss: 0.07730676233768463 MSE1: 0.006004407 MAE1: 0.051828377 R21: 0.548443911938516 MSE2: 0.00041701453 MAE2: 0.015657762 R22: 0.6870895974825267
[Test]: 145it [00:00, 264.12it/s]
lossAverage: 0.162769743280409 loss: 0.17872819304466248 MSE1: 0.0047573387 MAE1: 0.051048152 R21: 0.6857967137095158 MSE2: 0.000485776 MAE2: 0.016481854 R22: 0.6798371909383906

[Test]: 1143it [00:03, 323.74it/s]
lossAverage: 0.0923438661999835 loss: 0.025225277990102768 MSE1: 0.0039406684 MAE1: 0.04754775 R21: 0.6739719437539446 MSE2: 0.00040116478 MAE2: 0.015166105 R22: 0.6726640615283241
[Test]: 143it [00:00, 242.08it/s]
lossAverage: 18.931702666691454 loss: 0.023543821647763252 MSE1: 0.005367821 MAE1: 0.048827633 R21: 0.5262524152892094 MSE2: 0.00037365992 MAE2: 0.01472859 R22: 0.6731768370168081
[Test]: 145it [00:00, 251.05it/s]
lossAverage: 0.14536525803920586 loss: 0.045745570212602615 MSE1: 0.0041760756 MAE1: 0.048400603 R21: 0.695618219236751 MSE2: 0.00044117437 MAE2: 0.015766878 R22: 0.6833104883186959

[Test]: 1143it [00:11, 97.63it/s]
lossAverage: 0.09869258985443013 loss: 0.05024847015738487 MSE1: 0.0039293724 MAE1: 0.04696609 R21: 0.6812669789915118 MSE2: 0.00039435606 MAE2: 0.0148863075 R22: 0.6716623047298911
[Test]: 143it [00:01, 86.39it/s]
lossAverage: 16.4477666478984 loss: 0.07217039912939072 MSE1: 0.004800687 MAE1: 0.04718089 R21: 0.6323304944795786 MSE2: 0.00031747206 MAE2: 0.013550832 R22: 0.7505691230363849
[Test]: 145it [00:01, 93.39it/s] 
lossAverage: 0.13754051594501263 loss: 0.14298340678215027 MSE1: 0.0038972513 MAE1: 0.048726037 R21: 0.7381380935843421 MSE2: 0.00039415396 MAE2: 0.015372715 R22: 0.7274994275848637



[Test]: 9152it [01:24, 108.36it/s]
lossAverage: 0.07813332093279433 loss: 0.0018888266058638692 MSE1: 0.00049015816 MAE1: 0.01628682 R21: 0.9616543961103535 MSE2: 5.4609027e-05 MAE2: 0.005334254 R22: 0.9839440400274879
[Test]: 1144it [00:10, 105.97it/s]
lossAverage: 0.7274209770122001 loss: 0.00013692631910089403 MSE1: 0.0018678819 MAE1: 0.032135896 R21: 0.857146683325386 MSE2: 0.000325814 MAE2: 0.010539407 R22: 0.8916914145577166
[Test]: 1152it [00:10, 104.73it/s]
lossAverage: 0.28536418158484733 loss: 0.04180380702018738 MSE1: 0.0016818182 MAE1: 0.03088642 R21: 0.8584327962612048 MSE2: 0.00018589142 MAE2: 0.009880457 R22: 0.845025123660823
[Test]: 9152it [01:25, 107.44it/s]
lossAverage: 0.08964535185841371 loss: 0.0018588766688480973 MSE1: 0.00050048216 MAE1: 0.016490582 R21: 0.961166794219738 MSE2: 5.5528788e-05 MAE2: 0.005403719 R22: 0.9837156071134396
[Test]: 1144it [00:10, 105.38it/s]
lossAverage: 0.8110854743355641 loss: 0.002390687819570303 MSE1: 0.0018617677 MAE1: 0.03213868 R21: 0.8588942719024671 MSE2: 0.00032692842 MAE2: 0.010543808 R22: 0.8917234407064543
[Test]: 1152it [00:10, 105.83it/s]
lossAverage: 0.3212995257498715 loss: 0.001952794031240046 MSE1: 0.0016791198 MAE1: 0.03089728 R21: 0.8598939573900415 MSE2: 0.00018718127 MAE2: 0.0098816715 R22: 0.8453806027644728
[Test]: 9152it [01:25, 107.60it/s]
lossAverage: 0.0904467972164046 loss: 0.0017648707143962383 MSE1: 0.0005015386 MAE1: 0.016509159 R21: 0.961096781967734 MSE2: 5.5626555e-05 MAE2: 0.0054098205 R22: 0.983687132026833
[Test]: 1144it [00:10, 104.96it/s]
lossAverage: 0.8147860602455647 loss: 0.0001072470040526241 MSE1: 0.0018625318 MAE1: 0.0321423 R21: 0.8588814839428914 MSE2: 0.00032727706 MAE2: 0.0105453385 R22: 0.8915947998085823
[Test]: 1152it [00:10, 105.97it/s]
lossAverage: 0.323816593444992 loss: 0.011026461608707905 MSE1: 0.0016790727 MAE1: 0.030897485 R21: 0.8599470372747026 MSE2: 0.00018730959 MAE2: 0.009882265 R22: 0.8453515060047839


[Test]: 9152it [01:15, 120.72it/s]
lossAverage: 0.00036163969376792833 loss: 8.60434738569893e-05 MSE1: 9.283792e-06 MAE1: 0.0016607399 R21: 0.9993058051176285 MSE2: 3.3337078e-06 MAE2: 0.0010809936 R22: 0.9990355305236819
[Test]: 1143it [00:09, 124.50it/s]
lossAverage: 0.12071574800478845 loss: 0.001227163360454142 MSE1: 0.0017239668 MAE1: 0.030675432 R21: 0.8699344892172409 MSE2: 0.00018123131 MAE2: 0.009850082 R22: 0.8612582136468634
[Test]: 1152it [00:09, 122.88it/s]
lossAverage: 0.0642365936121251 loss: 0.12574800848960876 MSE1: 0.0016868466 MAE1: 0.030766705 R21: 0.8629221188862577 MSE2: 0.00017553449 MAE2: 0.00985058 R22: 0.8561952185863201
[Test]: 9152it [01:13, 124.01it/s]
lossAverage: 0.01710669895149035 loss: 0.0006652774172835052 MSE1: 0.0002611414 MAE1: 0.011570367 R21: 0.9791815918043401 MSE2: 3.3279266e-05 MAE2: 0.0040244358 R22: 0.9904282356114014
[Test]: 1143it [00:09, 122.90it/s]
lossAverage: 0.17641790963699533 loss: 0.04875212907791138 MSE1: 0.0017767499 MAE1: 0.030822804 R21: 0.8568317446627264 MSE2: 0.00018957155 MAE2: 0.010009632 R22: 0.8449860636011919
[Test]: 1152it [00:09, 123.63it/s]
lossAverage: 0.07751759945793937 loss: 1.5833329598535784e-05 MSE1: 0.0018139989 MAE1: 0.03254018 R21: 0.8405879588851974 MSE2: 0.00018583541 MAE2: 0.010323042 R22: 0.8350342145229603
[Test]: 9152it [01:13, 124.27it/s]
lossAverage: 0.0010087537594498542 loss: 0.0009917281568050385 MSE1: 2.2420094e-05 MAE1: 0.0030721428 R21: 0.9983464583697768 MSE2: 5.2580317e-06 MAE2: 0.0014766544 R22: 0.9984909690600164
[Test]: 1143it [00:09, 123.38it/s]
lossAverage: 0.13720804728730504 loss: 0.24921788275241852 MSE1: 0.0017123263 MAE1: 0.030639552 R21: 0.8722460116171512 MSE2: 0.00018063962 MAE2: 0.009872981 R22: 0.8636666244173066
[Test]: 1152it [00:09, 124.71it/s]
lossAverage: 0.07264397032813205 loss: 0.006352441385388374 MSE1: 0.001686726 MAE1: 0.030938625 R21: 0.8644949594060564 MSE2: 0.00017461597 MAE2: 0.009860087 R22: 0.8587801650372362


[Test]: 9152it [01:10, 129.00it/s]
lossAverage: 0.006448110825189934 loss: 0.0007106066332198679 MSE1: 0.0003039526 MAE1: 0.010915979 R21: 0.9765525922305573 MSE2: 3.3403972e-05 MAE2: 0.003819561 R22: 0.99024283938272
[Test]: 1143it [00:08, 133.50it/s]
lossAverage: 0.053416719951617835 loss: 0.006228163372725248 MSE1: 0.0014534673 MAE1: 0.02675305 R21: 0.8875746118194251 MSE2: 0.00014771354 MAE2: 0.008630112 R22: 0.885186717313915
[Test]: 1152it [00:08, 134.16it/s]
lossAverage: 0.030233841569136014 loss: 0.0038444276433438063 MSE1: 0.0012128914 MAE1: 0.024889816 R21: 0.9006967316439484 MSE2: 0.0001235679 MAE2: 0.008033821 R22: 0.8986985147761174
[Test]: 9152it [01:09, 131.51it/s]
lossAverage: 0.007923301152369035 loss: 0.003462323686107993 MSE1: 0.00035850078 MAE1: 0.012110153 R21: 0.9731843986694118 MSE2: 3.883027e-05 MAE2: 0.00416164 R22: 0.9887476673521192
[Test]: 1143it [00:08, 128.02it/s]
lossAverage: 0.057408323435270324 loss: 0.03042222186923027 MSE1: 0.0014898331 MAE1: 0.027198033 R21: 0.8881953171992636 MSE2: 0.0001513879 MAE2: 0.008757291 R22: 0.886110646396457
[Test]: 1152it [00:09, 127.28it/s]
lossAverage: 0.03690072052867365 loss: 0.006283373571932316 MSE1: 0.0012537531 MAE1: 0.02527877 R21: 0.9007738282302497 MSE2: 0.00012826236 MAE2: 0.008120898 R22: 0.8984916870057322
[Test]: 9152it [01:11, 128.76it/s]
lossAverage: 0.007256276999610122 loss: 0.0006813873187638819 MSE1: 0.0003335984 MAE1: 0.011548007 R21: 0.9743800993986367 MSE2: 3.6444722e-05 MAE2: 0.0040133535 R22: 0.9893281320153845
[Test]: 1143it [00:08, 127.77it/s]
lossAverage: 0.05045391041658108 loss: 0.2617013454437256 MSE1: 0.0014610556 MAE1: 0.026923986 R21: 0.8873777133987871 MSE2: 0.0001484558 MAE2: 0.0086706085 R22: 0.8850820056894888
[Test]: 1152it [00:08, 128.92it/s]
lossAverage: 0.03005943621527813 loss: 0.016473812982439995 MSE1: 0.0012115272 MAE1: 0.024977062 R21: 0.9013016230541692 MSE2: 0.0001237551 MAE2: 0.008057365 R22: 0.8991330497159777


[Test]: 9152it [01:09, 130.86it/s]
lossAverage: 0.29667331237239947 loss: 6.257975473999977e-05 MSE1: 0.0005257475 MAE1: 0.014664048 R21: 0.9586022526870236 MSE2: 5.3817148e-05 MAE2: 0.004820664 R22: 0.9841671292454086
[Test]: 1143it [00:09, 126.91it/s]
lossAverage: 1.0754202711554621 loss: 0.0013241404667496681 MSE1: 0.0010225194 MAE1: 0.021651972 R21: 0.9214688852803591 MSE2: 0.00010070114 MAE2: 0.0069227084 R22: 0.9234965633275358
[Test]: 1152it [00:09, 127.93it/s]
lossAverage: 0.6198674000179475 loss: 0.05316803231835365 MSE1: 0.0008680516 MAE1: 0.019707285 R21: 0.926605416147604 MSE2: 8.719827e-05 MAE2: 0.006302736 R22: 0.9269543818204953
[Test]: 9152it [01:11, 128.19it/s]
lossAverage: 0.29692602566018717 loss: 0.005339266266673803 MSE1: 0.0005258151 MAE1: 0.014665392 R21: 0.9586059108144949 MSE2: 5.3820728e-05 MAE2: 0.00482083 R22: 0.9841668371944265
[Test]: 1143it [00:09, 124.67it/s]
lossAverage: 1.0767886472160624 loss: 0.011017384938895702 MSE1: 0.0010226078 MAE1: 0.02165309 R21: 0.9214799868515204 MSE2: 0.00010068949 MAE2: 0.006922956 R22: 0.9235231616017114
[Test]: 1152it [00:09, 122.92it/s]
lossAverage: 0.620661708175482 loss: 0.002749746199697256 MSE1: 0.0008680922 MAE1: 0.019707037 R21: 0.9266162425509287 MSE2: 8.7191474e-05 MAE2: 0.006302076 R22: 0.9269739847161818
[Test]: 9152it [01:13, 124.67it/s]
lossAverage: 0.30621968023467416 loss: 0.0003913328400813043 MSE1: 0.0005324318 MAE1: 0.014789091 R21: 0.9582203583049589 MSE2: 5.4361106e-05 MAE2: 0.004853644 R22: 0.9840330593841343
[Test]: 1143it [00:09, 126.49it/s]
lossAverage: 1.1083569527620603 loss: 0.030608588829636574 MSE1: 0.0010304217 MAE1: 0.021742817 R21: 0.9210962686911505 MSE2: 0.0001009215 MAE2: 0.006944144 R22: 0.9235330219966952
[Test]: 1152it [00:09, 124.65it/s]
lossAverage: 0.6380391622376007 loss: 0.08589477837085724 MSE1: 0.0008708869 MAE1: 0.01975882 R21: 0.9266222853603375 MSE2: 8.721718e-05 MAE2: 0.0063096145 R22: 0.9271688597967374


[Test]: 9152it [01:06, 137.58it/s]
lossAverage: 0.00063587696378782 loss: 1.994794547499623e-05 MSE1: 3.1031486e-05 MAE1: 0.0036280488 R21: 0.9976729619744992 MSE2: 6.157653e-06 MAE2: 0.0014304583 R22: 0.9982170346236716
[Test]: 1143it [00:08, 141.45it/s]
lossAverage: 0.00485144279887401 loss: 1.7307334928773344e-05 MSE1: 0.00022345349 MAE1: 0.009372313 R21: 0.9838529324695312 MSE2: 2.6468642e-05 MAE2: 0.0031764559 R22: 0.980684961029031
[Test]: 1152it [00:08, 137.73it/s]
lossAverage: 0.004306836486391383 loss: 0.00020253051479812711 MSE1: 0.00021466399 MAE1: 0.009769779 R21: 0.9828264590218083 MSE2: 2.8366041e-05 MAE2: 0.003295362 R22: 0.9772211850798911
[Test]: 9152it [01:07, 136.50it/s]
lossAverage: 0.0006380521813388825 loss: 5.8472978707868606e-05 MSE1: 3.1135518e-05 MAE1: 0.0036346016 R21: 0.9976647720662852 MSE2: 6.167351e-06 MAE2: 0.0014326556 R22: 0.9982140613876442
[Test]: 1143it [00:08, 139.56it/s]
lossAverage: 0.0048175279652761535 loss: 0.0009230076684616506 MSE1: 0.00022331024 MAE1: 0.00937409 R21: 0.983859362449597 MSE2: 2.6451295e-05 MAE2: 0.0031768258 R22: 0.9806940935351015
[Test]: 1152it [00:08, 140.97it/s]
lossAverage: 0.004298403976920199 loss: 7.62845593271777e-05 MSE1: 0.00021488112 MAE1: 0.009773878 R21: 0.982803718284205 MSE2: 2.837956e-05 MAE2: 0.0032968028 R22: 0.9772050768329433
[Test]: 9152it [01:07, 136.30it/s]
lossAverage: 0.0006368699688010807 loss: 0.00010545273107709363 MSE1: 3.1047213e-05 MAE1: 0.0036287988 R21: 0.997671827150214 MSE2: 6.158973e-06 MAE2: 0.0014307015 R22: 0.9982166984898656
[Test]: 1143it [00:08, 140.78it/s]
lossAverage: 0.0048485909291312536 loss: 0.004413940478116274 MSE1: 0.00022342501 MAE1: 0.00937358 R21: 0.9838554089245433 MSE2: 2.6462738e-05 MAE2: 0.003176105 R22: 0.9806904957992139
[Test]: 1152it [00:08, 137.19it/s]
lossAverage: 0.004307965088616843 loss: 0.007963530719280243 MSE1: 0.00021464362 MAE1: 0.009769085 R21: 0.9828284394777042 MSE2: 2.8361734e-05 MAE2: 0.0032950244 R22: 0.9772259403519772
'''

