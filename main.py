from config import *
from datetime import datetime
from basenetwork import *
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd


from EEGIFNet import *

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def data_prep(batch_size):
    path = '/data2/ch/EEGdenoise/data/dataset/dataset2/'
    train_input = np.load(path + 'EMG_train_input.npy')
    train_output = np.load(path + 'EMG_train_output.npy')
    val_input = np.load(path + 'EMG_val_input.npy')
    val_output = np.load(path + 'EMG_val_output.npy')
    test_input = np.load(path + 'EMG_test_input.npy')
    test_output = np.load(path + 'EMG_test_output.npy')
    # train_input = np.load(path + 'EMG_train_input_512hz.npy')
    # train_output = np.load(path + 'EMG_train_output_512hz.npy')
    # val_input = np.load(path + 'EMG_val_input_512hz.npy')
    # val_output = np.load(path + 'EMG_val_output_512hz.npy')
    # test_input = np.load(path + 'EMG_test_input_512hz.npy')
    # test_output = np.load(path + 'EMG_test_output_512hz.npy')
    trainset = my_dataset(train_input, train_output)
    valset = my_dataset(val_input, val_output)
    testset = my_dataset(test_input, test_output)
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=True)
    # test_dataloader = DataLoader(testset, batch_size=340, shuffle=False) #E O G
    # test_dataloader = DataLoader(testset, batch_size=560, shuffle=False) #E M G
    # test_dataloader = DataLoader(testset, batch_size=276, shuffle=False) #m o t i o n
    # test_dataloader = DataLoader(testset, batch_size=360, shuffle=False)  # P L I & E C G
    # test_dataloader = DataLoader(testset, batch_size=141, shuffle=False)  # dataset2 E O G
    test_dataloader = DataLoader(testset, batch_size=340, shuffle=False)  # dataset2 E M G
    return train_dataloader, val_dataloader, test_dataloader


def train(I, M, device, train_dataloader, val_dataloader, epochs, learning_rate, bn_params):

    optimizer_I = torch.optim.RMSprop(I.parameters(), lr=learning_rate, alpha=0.9)
    optimizer_M = torch.optim.RMSprop(M.parameters(), lr=learning_rate, alpha=0.9)

    criterion = nn.MSELoss()
    criterion1 = nn.MSELoss()
    criterion2 = nn.MSELoss()
    #criterion = nn.L1Loss()
    #criterion = nn.SmoothL1Loss()
    best_val_loss = 200.0

    for epoch in range(epochs):

        # 训练步骤
        total_train_loss_e_per_epoch = 0
        total_train_loss_n_per_epoch = 0
        total_train_loss_per_epoch = 0
        train_step_num = 0
        I.train()
        M.train()

        for batch_idx, data in enumerate(train_dataloader):
            train_step_num += 1
            x, y = data
            x = x.float().to(device)
            y = y.float().to(device)
            z = x.squeeze()-y
            z = z.detach()

            optimizer_I.zero_grad()  # clear up the grads of optimized Variable
            optimizer_M.zero_grad()
            e_outputs, n_outputs = I(x)
            outputs = M(x, e_outputs, n_outputs)
            loss_e = criterion(e_outputs, y)
            loss_n = criterion1(n_outputs, z)
            loss_all = criterion(outputs, y)
            total_train_loss_e_per_epoch = total_train_loss_e_per_epoch + loss_e.item()
            total_train_loss_n_per_epoch = total_train_loss_n_per_epoch + loss_n.item()
            total_train_loss_per_epoch = total_train_loss_per_epoch + loss_all.item()

            #slim_loss = 1e-3 * sum([slim_penalty(m).to(device) for m in bn_params])

            loss = loss_e+loss_n+loss_all
            #loss = (loss_n) * (epoch < 10) + (loss_n + loss_all) * (epoch >= 10)
            loss.backward()
            optimizer_I.step()
            optimizer_M.step()


            #loss = (loss_e + loss_n)*(epoch < 40) + (0.1*loss_e + 0.1*loss_n +loss_all)*(epoch >= 40)
            # if epoch < 40:
            #     loss = loss_e + loss_n
            #     # loss = loss_n + loss_all
            #     loss.backward()
            #     optimizer_I.step()
            # elif epoch >= 40 and epoch < 70:
            #     loss = loss_all
            #     loss.backward()
            #     optimizer_M.step()
            # else:
            #     loss = loss_all
            #     loss.backward()
            #     optimizer_I.step()
            #     optimizer_M.step()
            optimizer_I.zero_grad()  # clear up the grads of optimized Variable
            optimizer_M.zero_grad()
        average_train_loss_e_per_epoch = total_train_loss_e_per_epoch / train_step_num
        average_train_loss_n_per_epoch = total_train_loss_n_per_epoch / train_step_num
        average_train_loss_per_epoch = total_train_loss_per_epoch / train_step_num
        print("epoch-{}/{} mse_e: {} mse_n: {} mse_all: {}\n".format(epoch + 1, epochs, average_train_loss_e_per_epoch, average_train_loss_n_per_epoch, average_train_loss_per_epoch))
        # with SummaryWriter("./log/"+"{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())) as writer:
        #     writer.add_scalar("train_loss", average_train_loss_per_epoch, global_step=epoch)


        # 验证步骤
        val_step_num = 0
        total_val_loss_per_epoch = 0
        sum_acc, sum_acc_n, sum_acc_e = 0,0,0
        sum_rrmse, sum_rrmse_n, sum_rrmse_e = 0,0,0

        I.eval()
        M.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(val_dataloader):
                val_step_num += 1

                x, y = data
                x = x.float().to(device)
                y = y.float().to(device)

                e_outputs, n_outputs = I(x)
                outputs = M(x, e_outputs, n_outputs)
                #outputs = e_outputs

                loss = criterion(outputs, y)
                #loss = criterion(outputs, x.squeeze() - y)
                total_val_loss_per_epoch += loss.item()

                # 计算ACC
                acc_n = cal_ACC_tensor(n_outputs.detach(), x.detach().squeeze() - y.detach())
                sum_acc_n += acc_n
                rrmse_n = cal_RRMSE_tensor(n_outputs.detach(), x.detach().squeeze() - y.detach())
                sum_rrmse_n += rrmse_n

                acc_e = cal_ACC_tensor(e_outputs.detach(), y.detach())
                sum_acc_e += acc_e
                rrmse_e = cal_RRMSE_tensor(e_outputs.detach(), y.detach())
                sum_rrmse_e += rrmse_e

                #acc = cal_ACC_tensor(outputs.detach(), x.detach().squeeze() - y.detach())
                acc = cal_ACC_tensor(outputs.detach(), y.detach())
                sum_acc += acc
                #rrmse = cal_RRMSE_tensor(outputs.detach(), x.detach().squeeze() - y.detach())
                rrmse = cal_RRMSE_tensor(outputs.detach(), y.detach())
                sum_rrmse += rrmse

            average_val_loss_per_epoch = total_val_loss_per_epoch / val_step_num
            acc = sum_acc.item() / val_step_num
            rrmse = sum_rrmse.item() / val_step_num

            acc_e= sum_acc_e.item() / val_step_num
            rrmse_e = sum_rrmse_e.item() / val_step_num

            acc_n = sum_acc_n.item() / val_step_num
            rrmse_n = sum_rrmse_n.item() / val_step_num

            print("[epoch %d/%d] [LOSS: %f] [ACC_e: %f] [RRMSE_e: %f]" % (
                epoch + 1, epochs, average_val_loss_per_epoch, acc_e, rrmse_e))
            print("[ACC_n: %f] [RRMSE_n: %f]" % (acc_n, rrmse_n))
            print("[ACC: %f] [RRMSE: %f]" % (acc, rrmse))
            # with SummaryWriter("./log/"+"{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())) as writer:
            #     writer.add_scalar("val_loss", average_val_loss_per_epoch, global_step=epoch)
            #     writer.add_scalar("ACC", acc, global_step=epoch)

            if average_val_loss_per_epoch < best_val_loss:
                print('save model')
                torch.save(I.state_dict(), 'checkpoint/EMG_INet2.pkl')
                torch.save(M.state_dict(), 'checkpoint/EMG_MNet2.pkl')
                best_val_loss = average_val_loss_per_epoch


def pplot(I,M,test_input,test_output):
    num = test_input.shape[0]
    test_input_tensor = torch.tensor(test_input).float().to(device)
    test_output_tensor = torch.tensor(test_output).float().to(device)
    I.eval()
    M.eval()
    with torch.no_grad():
        predict_EEG, predict_A = I(test_input_tensor)
        outputs = M(test_input_tensor, predict_EEG, predict_A)
    #print(predict_test.shape)
    #np.save('./checkpoint/predict_test.npy', predict_test)
    # calculate ACC
    # acc = cal_ACC_tensor(predict_test.detach(), test_output_tensor.detach()).item()
    # print("ACC = " + str(acc))
    test_input_tensor = test_input_tensor.squeeze()
    predict_EEG2 = test_input_tensor - predict_A

    predict_EEG = predict_EEG.cpu()
    predict_EEG2 = predict_EEG2.cpu()
    outputs = outputs.cpu()

    for i in range(num):
        plt.subplot(311)
        l0, = plt.plot(test_input[i].squeeze())
        l1, = plt.plot(predict_EEG[i])
        l2, = plt.plot(test_output[i])
        #plt.legend([l0, l1, l2], ['Raw EEG', 'Denoised EEG', 'Clean EEG'], loc='upper center')
        # plt.legend([l0, l1, l2], ['Raw EEG', 'Denoised EEG', 'Clean EEG'], loc='upper center', bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
        plt.legend([l0, l1, l2], ['Contaminated EEG', 'Denoised EEG', 'Clean EEG'], loc='upper center', bbox_to_anchor=(0.5, 1.3),
                   fancybox=False, shadow=False, ncol=3)
        plt. xticks([])
        plt.title('(1)',y=-0.2)
        plt.subplot(312)
        l0, = plt.plot(test_input[i].squeeze())
        l1, = plt.plot(predict_EEG2[i])
        l2, = plt.plot(test_output[i])
        #plt.legend([l0, l1, l2], ['Raw EEG', 'Denoised EEG', 'Clean EEG'], loc='upper right')
        plt.xticks([])
        plt.title('(2)',y=-0.2)
        plt.subplot(313)
        l0, = plt.plot(test_input[i].squeeze())
        l1, = plt.plot(outputs[i])
        l2, = plt.plot(test_output[i])
        plt.title('(3)',y=-0.3)
        #plt.legend([l0, l1, l2], ['Raw EEG', 'Denoised EEG', 'Clean EEG'], loc='upper right')
        plt.savefig('./checkpoint/EMG_'+str(i)+'.png',dpi = 300)
        plt.close()


def test(I,M, test_dataloader):
    I.eval()
    M.eval()

    val_step_num = 0
    acc_list =[]
    rrmse_list =[]
    snr_list = []
    sum_acc = 0
    sum_rrmse = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_dataloader):
            val_step_num += 1

            x, y = data
            x = x.float().to(device)
            y = y.float().to(device)

            e_outputs, n_outputs = I(x)
            outputs = M(x, e_outputs, n_outputs)

            #outputs = e_outputs
            #outputs = x.squeeze() - n_outputs

            # 计算ACC
            #acc = cal_ACC_tensor(outputs.detach(), x.detach().squeeze() - y.detach()).item()
            acc = cal_ACC_tensor(outputs.detach(), y.detach()).item()
            acc_list.append(acc)
            #rrmse = cal_RRMSE_tensor(outputs.detach(), x.detach().squeeze() - y.detach()).item()
            rrmse = cal_RRMSE_tensor(outputs.detach(), y.detach()).item()
            rrmse_list.append(rrmse)

            snr = cal_SNR(outputs, y)
            snr_list.append(snr)

        acc_all = np.array(acc_list)
        rrmse_all = np.array(rrmse_list)
        acc = np.mean(acc_all)
        rrmse = np.mean(rrmse_all)
        print(acc_all)
        print("acc = " + str(acc))

        print(rrmse_all)
        print("rrmse = " + str(rrmse))
        snr_all = np.array(snr_list)
        snr = np.mean(snr_all)
        print(snr_all)
        print("snr = " + str(snr))

        return acc_list, rrmse_list, snr_list


if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    batch_size = 256
    epochs = 80
    learning_rate = 5e-5
    train_dataloader, val_dataloader, test_dataloader = data_prep(batch_size)



    I = MA_INet().apply(weights_init).to(device)
    M = MA_MNet().apply(weights_init).to(device)
    #
    # slim_penalty = lambda var: torch.abs(var).sum()
    #
    # slim_params, bn_params = [], []
    # for name, param in I.named_parameters():
    #     if param.requires_grad and name.endswith('weight') and 'batnorm' in name:
    #         bn_params.append(param[len(param) // 2:])
    #         # if len(slim_params) % 2 == 0:
    #         #     slim_params.append(param[:len(param) // 2])
    #         # else:
    #         #     slim_params.append(param[len(param) // 2:])
    #
    # train(I, M, device, train_dataloader, val_dataloader, epochs, learning_rate, bn_params)
    #
    # print('----------------------------------------')
    # del I
    # del M

    I = MA_INet().to(device)
    I.load_state_dict(torch.load('./checkpoint/EMG_INet3.pkl'))
    M = MA_MNet().to(device)
    M.load_state_dict(torch.load('./checkpoint/EMG_MNet3.pkl'))
    acc, rrmse, snr = test(I,M, test_dataloader)

    acclist = pd.DataFrame(acc)
    acclist.to_csv("./result/EMG_IMNet2_acc" + ".csv")
    rrmselist = pd.DataFrame(rrmse)
    rrmselist.to_csv("./result/EMG_IMNet2_rrmse" + ".csv")
    snrlist = pd.DataFrame(snr)
    snrlist.to_csv("./result/EMG_IMNet2_snr" + ".csv")


    # test_num = test_input.shape[0]//10
    # acclist=[]
    # for i in range(10):
    #     acc, predict_test = test(model, test_input[test_num*i:test_num*(i+1)], test_output[test_num*i:test_num*(i+1)])
    #     acclist.append(acc)
    # print(acclist)
    # acclist = pd.DataFrame(acclist)
    # acclist.to_csv("acc_all" + ".csv")

    #choose one sample for visualization
    # index = 360*0+0
    # test_input = np.load('/data2/ch/EEGdenoise/data/EMG_test_input.npy')
    # test_output = np.load('/data2/ch/EEGdenoise/data/EMG_test_output.npy')
    # test_input = test_input.reshape(-1,1,test_input.shape[-1])
    # pplot(I,M, test_input[index:index+250], test_output[index:index+250])



    # acc, predict_test = test(model, test_input[index:index+2], test_output[index:index+2])
    # l0, = plt.plot(test_input.squeeze()[index])
    # l1, = plt.plot(predict_test[0])
    # l2, = plt.plot(test_output[index])
    #
    # plt.legend([l0, l1, l2], ['Raw EEG', 'Denoised EEG', 'Clean EEG'], loc='upper right')
    # plt.savefig('./checkpoint/1.png',dpi=300)
    # plt.show()
