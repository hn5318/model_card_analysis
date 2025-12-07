# encoding=utf-8
import os
import sys
import json

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

import pandas as pd
import logger
import glo
from dl_models import *
from utils import *
import re
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")


def save_model(epoch, model, training_stats, info, model_name):
    base_dir = './modelcard_'+ info + '_'+ model_name + '/epoch_' + str(epoch) + '/'
    out_dir = base_dir + 'model.ckpt'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    print('Saving model to %s' % out_dir)
    torch.save(model.state_dict(), out_dir)
    df_stats = pd.DataFrame(data=training_stats)
    df_stats.to_json(base_dir + "training_stats.json")


def dl_container(modelcard_data, train_y, test_modelcard_data, test_y, logger, info, fold_num, model_type, use_imbalanced_sampler=True):
    config = Config()
    if model_type == "bi-lstm":
        model_name = "bi-lstm"
        model = bilstm_model(config).to(config.device)
    elif model_type == "textcnn":
        model_name = "textcnn"
        model = cnn_model(config).to(config.device)
    elif model_type == "transformer":
        model_name = "transformer"
        model = bert_trans_model(config).to(config.device)
    
    # 只在CUDA可用且有多个GPU时使用DataParallel
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
    else:
        print(f"使用 {config.device} 进行训练")

    logger.info("============begin %s training......" % model_name)
    # 训练数据使用不平衡采样器（如果启用）
    train_dt = Data_processor(modelcard_data, train_y, config.batch_size, 
                             use_imbalanced_sampler=use_imbalanced_sampler, is_training=True)
    traindata_loader = train_dt.processed_dataloader
    # 测试数据不使用采样器，按顺序处理
    test_dt = Data_processor(test_modelcard_data, test_y, config.batch_size, 
                            use_imbalanced_sampler=False, is_training=False)
    testdata_loader = test_dt.processed_dataloader
    print("model created.")
    epochs = config.num_epochs
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(traindata_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    seed_val = 3407
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    training_stats = []
    total_t0 = time.time()
    model.eval()
    loss_fn = F.cross_entropy
    best_epoch_f1 = 0.0  # 改为追踪最佳F1分数
    print("Begin training...")
    progress_bar = tqdm(range(total_steps))
    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        t0 = time.time()
        epcho_train_loss = 0
        model.train()
        for step, batch in enumerate(traindata_loader):
            if step % 100 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(traindata_loader), elapsed))

            input_ids = batch[0].to(config.device)
            input_mask = batch[1].to(config.device)
            batch_labels = batch[2].to(config.device)

            model.zero_grad()
            modelcard_input = (input_ids, input_mask)
            batch_outputs = model(modelcard_input)
            loss = loss_fn(batch_outputs, batch_labels)
            epcho_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        avg_train_loss = epcho_train_loss / len(traindata_loader)
        training_time = format_time(time.time() - t0)
        print("")
        print("====== Average training loss: {0:.2f}".format(avg_train_loss))
        print("====== Training epcoh took: {:}".format(training_time))
        print("Running Testing....")
        t0 = time.time()
        model.eval()
        prob_result_lst = []
        all_pre_label = []
        total_eval_accuracy = 0
        total_eval_loss = 0
        for batch in testdata_loader:
            with torch.no_grad():
                input_ids = batch[0].to(config.device)
                input_mask = batch[1].to(config.device)
                batch_labels = batch[2].to(config.device)
                modelcard_input = (input_ids, input_mask)
                b_outputs = model(modelcard_input)
            loss = loss_fn(b_outputs, batch_labels)
            total_eval_loss += loss.item()
            lr = torch.nn.Softmax(dim=1)
            preds_prob = lr(b_outputs.data)
            preds_prob = preds_prob.cpu().detach().numpy()
            prob_result_lst.append(preds_prob)
            preds = torch.max(b_outputs.data, 1)[1].cpu().numpy()
            labels = batch_labels.to('cpu').numpy()
            all_pre_label.extend(preds.flatten())
            total_eval_accuracy += flat_accuracy(preds, labels)
        prob_result = np.vstack(prob_result_lst)
        print("prob_result:", type(prob_result), prob_result.shape)
        
        # 打印预测情况统计
        pred_counts = np.bincount(all_pre_label)
        test_counts = np.bincount(test_y)
        print(f"测试集标签分布: 负类={test_counts[0]}, 正类={test_counts[1] if len(test_counts) > 1 else 0}")
        print(f"预测结果分布: 负类={pred_counts[0]}, 正类={pred_counts[1] if len(pred_counts) > 1 else 0}")

        avg_val_accuracy = total_eval_accuracy / len(testdata_loader)
        print("  Accuracy: {0:.3f}".format(avg_val_accuracy))
        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(testdata_loader)
        test_time = format_time(time.time() - t0)
        print("  Validation Loss: {0:.3f}".format(avg_val_loss))
        print("  Validation took: {:}".format(test_time))
        training_stats.append(
            {'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': test_time
            })
        # evulates each epohc, save the metrics with the best accuracy.
        acc_res = accuracy_score(test_y, all_pre_label)
        recall_res = recall_score(test_y, all_pre_label, average=None)
        precision_res = precision_score(test_y, all_pre_label, average=None)
        f1_res = f1_score(test_y, all_pre_label, average=None)
        
        # 输出详细的验证指标
        print(f"\n====== Epoch {epoch_i + 1} 验证结果 ======")
        print(f"准确率: {acc_res:.4f}")
        print(f"精确率 (正/负): {precision_res[1]:.4f} / {precision_res[0]:.4f}")
        print(f"召回率 (正/负): {recall_res[1]:.4f} / {recall_res[0]:.4f}")
        print(f"F1分数 (正/负): {f1_res[1]:.4f} / {f1_res[0]:.4f}")
        
        y_score = prob_result[:, 1]
        aucScore = average_precision_score(y_true=test_y, y_score=y_score)
        rocAucScore = roc_auc_score(y_true=test_y, y_score=y_score)
        
        metrics_data = [precision_res[1], precision_res[0], acc_res, 1, recall_res[1], recall_res[0], f1_res[0], f1_res[1], aucScore, rocAucScore]
        
        glo_key = model_name + '_' + info + '_' + str(fold_num)
        # 使用正类F1分数作为保存模型的标准
        current_f1 = f1_res[1]  # 正类（类别1）的F1分数
        if current_f1 >= best_epoch_f1:
            best_epoch_f1 = current_f1
            print(f"\n*** 新的最佳F1分数: {current_f1:.4f} (Epoch {epoch_i + 1}) ***")
            print(f"    准确率: {acc_res:.4f}, 召回率: {recall_res[1]:.4f}, 精确率: {precision_res[1]:.4f}")
            glo.set_val(glo_key, metrics_data)
            save_model(epoch_i + 1, model, training_stats, info, model_name)
        else:
            print(f"    当前F1分数: {current_f1:.4f} (最佳: {best_epoch_f1:.4f})")
    logger.info("============all epoches of %s trained finished......" % model_name)
    return


def cal_last_metrics(logger, model_name, info):

    keylist = [model_name + '_' + info + '_' + str(fold_num) for fold_num in range(0,10)]
    
    # 创建字典来保存每个fold的详细指标 
    fold_metrics = {}
    overall_results = {}
    results_per_fold = {}
    
    for k in keylist:
        metrics = glo.get_val(k)
        if metrics is None:
            continue
            
        fold_num = int(k.split('_')[-1])
        
        # 保存每个fold的详细指标
        fold_metrics[f'fold_{fold_num}'] = {
            'accuracy': float(metrics[2]),
            'precision_positive': float(metrics[0]),
            'precision_negative': float(metrics[1]),
            'recall_positive': float(metrics[4]),
            'recall_negative': float(metrics[5]),
            'f1_positive': float(metrics[7]),
            'f1_negative': float(metrics[6]),
            'pr_auc': float(metrics[8]),
            'roc_auc': float(metrics[9])
        }
        
        # 用于计算平均值
        if f'fold_{fold_num}' not in results_per_fold:
            results_per_fold[f'fold_{fold_num}'] = fold_metrics[f'fold_{fold_num}']
    
    # 计算平均指标
    if results_per_fold:
        num_folds = len(results_per_fold)
        avg_metrics = {
            'accuracy': sum([v['accuracy'] for v in results_per_fold.values()]) / num_folds,
            'precision_positive': sum([v['precision_positive'] for v in results_per_fold.values()]) / num_folds,
            'precision_negative': sum([v['precision_negative'] for v in results_per_fold.values()]) / num_folds,
            'recall_positive': sum([v['recall_positive'] for v in results_per_fold.values()]) / num_folds,
            'recall_negative': sum([v['recall_negative'] for v in results_per_fold.values()]) / num_folds,
            'f1_positive': sum([v['f1_positive'] for v in results_per_fold.values()]) / num_folds,
            'f1_negative': sum([v['f1_negative'] for v in results_per_fold.values()]) / num_folds,
            'pr_auc': sum([v['pr_auc'] for v in results_per_fold.values()]) / num_folds,
            'roc_auc': sum([v['roc_auc'] for v in results_per_fold.values()]) / num_folds
        }
        
        overall_results = avg_metrics
        
        # 保存结果到文件（与ml_classifiers.py格式一致）
        output_dir = f'./Deep_Learning_Models/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(f'{output_dir}overall_results.json', 'w') as f:
            json.dump(overall_results, f, indent=4)
        
        with open(f'{output_dir}results_per_fold.json', 'w') as f:
            json.dump(results_per_fold, f, indent=4)
        
        logger.info(f"Results saved to {output_dir}")
        
        # 输出结果信息
        info_res = f"\n fold {num_folds}" \
                  f"\tAccuracyMean: {avg_metrics['accuracy']:.4f}" \
                  f"\tPrecisionMean: {avg_metrics['precision_positive']:.4f}" \
                  f"\tNegativePrecisionMean: {avg_metrics['precision_negative']:.4f}" \
                  f"\tRecallMean: {avg_metrics['recall_positive']:.4f}" \
                  f"\tNegativeRecallMean: {avg_metrics['recall_negative']:.4f}" \
                  f"\tNegativeF1Mean: {avg_metrics['f1_negative']:.4f}" \
                  f"\tF1Mean: {avg_metrics['f1_positive']:.4f}" \
                  f"\tPR-AUCMean: {avg_metrics['pr_auc']:.4f}" \
                  f"\tROC-AUCMean: {avg_metrics['roc_auc']:.4f}"
        
        logger.info(info_res)
        print(info_res)


def load_all_data(file_path):
    data = pd.read_json(file_path, orient='records', lines=True)
    data = data.iloc[:758,:5]
    return data


def tmp_label_process(labeled_data):
    high_low_labels = labeled_data['quality'].apply(lambda x:1 if x == 1 else 0)
    return high_low_labels


def main(file_path, model_type, use_imbalanced_sampler=True):
    mylogger = logger.mylog("dl")
    labeled_data = load_all_data(file_path)
    high_low_labels = tmp_label_process(labeled_data)
    pos_num = np.sum(high_low_labels == 1)
    neg_num = np.sum(high_low_labels == 0)
    print(f"数据集统计: 正样本={pos_num}, 负样本={neg_num}, 比例={pos_num/neg_num:.3f}")
    
    if use_imbalanced_sampler:
        print("将使用 ImbalancedDatasetSampler 处理数据不平衡问题")
    else:
        print("使用标准的数据加载方式")
    
    glo._init()
    fold_num = 0
    # kf = KFold(n_splits=10, random_state=3408, shuffle=True)
    kf = StratifiedKFold(n_splits=10, random_state=3408, shuffle=True)
    for train_index, test_index in kf.split(labeled_data, high_low_labels):
        train_x, train_y = list(labeled_data.iloc[train_index, 4]), list(high_low_labels.iloc[train_index])
        test_x, test_y = list(labeled_data.iloc[test_index, 4]), list(high_low_labels.iloc[test_index])

        dl_container(train_x, train_y, test_x, test_y, mylogger, "score", fold_num, model_type, use_imbalanced_sampler)
        fold_num += 1

    # calculate 10-fold metrics at last
    mylogger.info("==================modelcard high_low classifers=============")
    cal_last_metrics(mylogger, model_type, "score")


if __name__ == '__main__':
    # file_path = "./modelcard_data (update).json"
    file_path = "/kaggle/working/dl_test/todo-classifier/modelcard_data (update).json"
    
    USE_IMBALANCED_SAMPLER = True  # 设置为 False 可禁用不平衡采样器
    
    # type 1: textcnn 2: bi-lstm 3: transformer
    # model_list = ["textcnn", "bi-lstm", "transformer"]
    model_list = ["textcnn"]
    for model_type in model_list:
        print(f"\n开始训练 {model_type} 模型...")
        main(file_path, model_type, use_imbalanced_sampler=USE_IMBALANCED_SAMPLER)
