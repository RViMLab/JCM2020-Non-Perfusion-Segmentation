
import os
from scipy import ndimage
import csv
import numpy as np

from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, precision_recall_curve

import matplotlib.pyplot as plt

def get_metrics(pred_folder, gt_folder, weights_folder, pixel_depth=255):
    print('Computing performance metrics...')
    precision_ar = []
    recall_ar = []
    fscore_ar = []
    files = os.listdir(pred_folder)
    csv_path = os.path.join(pred_folder, 'metrics.csv')
    with open(csv_path, mode='w') as metrics_file:
        metrics_writer = csv.writer(metrics_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for f in files:
            try:
                print(f)
                filename, file_ext = os.path.splitext(f)
                gt_path = os.path.join(os.path.join(gt_folder), f)
                weight_path = os.path.join(os.path.join(weights_folder), filename + '.png') # valid
                pred_path = os.path.join(os.path.join(pred_folder), filename + '.png')
                gt = (ndimage.imread(gt_path).astype(float))
                pred = (ndimage.imread(pred_path).astype(float))
                weight_img = (ndimage.imread(weight_path).astype(float))

                # img_rs = (np.reshape(img, [1, -1])/pixel_depth).astype(int)
                # gt_rs = (np.reshape(gt, [1, -1])/pixel_depth).astype(int)
                # weight_img_rs = (np.reshape(weight_img, [1, -1])/pixel_depth).astype(int)

                # gt_rs = (gt.flatten() / pixel_depth).astype(int)
                pred_rs = (pred.flatten() / pixel_depth).astype(int)
                # weight_img_rs = (weight_img.flatten() / pixel_depth).astype(int)
                gt_rs = (gt > (pixel_depth // 2)).astype(int).flatten()
                weight_img_rs = (weight_img > (pixel_depth // 2)).astype(int).flatten()

                precision, recall, fscore, _ = precision_recall_fscore_support(
                    gt_rs,
                    pred_rs,
                    sample_weight=weight_img_rs,
                    pos_label=1,
                    average='binary')

                precision_ar.append(precision)
                recall_ar.append(recall)
                fscore_ar.append(fscore)

                metrics_writer.writerow([f, precision, recall, fscore])
            except IOError as e:
                print('Could not process data.\nError', e, '- Skipping file.')
        metrics_writer.writerow(['avg', np.mean(precision_ar), np.mean(recall_ar), np.mean(fscore_ar)])
        metrics_writer.writerow(['std', np.std(precision_ar), np.std(recall_ar), np.std(fscore_ar)])

    print('Performance metrics saved...')


def get_fold_scores(pred_folder, gt_folder, weights_folder, cv_folder=None, pixel_depth=255):
    print('Computing ROC curve...')

    gt_scores = np.array([])
    pred_scores = np.array([])

    if cv_folder is None:
        files = os.listdir(pred_folder)
    else:
        files = os.listdir(cv_folder)

    csv_path = os.path.join(pred_folder, 'metrics_roc.csv')
    with open(csv_path, mode='w') as metrics_file:
        metrics_writer = csv.writer(metrics_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for f in files:
            try:
                print(f)
                filename, file_ext = os.path.splitext(f)
                gt_path = os.path.join(os.path.join(gt_folder), f)
                weight_path = os.path.join(os.path.join(weights_folder), filename + '.png')  # valid
                pred_path = os.path.join(os.path.join(pred_folder), filename + '.npy')
                gt = (ndimage.imread(gt_path).astype(float))
                # pred = (ndimage.imread(pred_path).astype(float))
                pred = np.load(pred_path)
                weight_img = (ndimage.imread(weight_path).astype(float))

                # gt_rs = (gt.flatten() / pixel_depth).astype(int)
                pred_rs = pred.flatten()
                # weight_img_rs = (weight_img.flatten() / pixel_depth).astype(int)
                gt_rs = (gt > (pixel_depth // 2)).astype(int).flatten()
                weight_img_rs = (weight_img > (pixel_depth // 2)).astype(int).flatten()

                valid_pred = pred_rs[np.where(weight_img_rs==1)]
                valid_gt = gt_rs[np.where(weight_img_rs==1)]

                # precision, recall, fscore, _ = precision_recall_fscore_support(
                #     gt_rs,
                #     pred_rs,
                #     sample_weight=weight_img_rs,
                #     pos_label=1,
                #     average='binary')

                # gt_scores.append(valid_gt)
                # pred_scores.append(valid_pred)
                gt_scores = np.append(gt_scores, valid_gt)
                pred_scores = np.append(pred_scores, valid_pred)

                # metrics_writer.writerow([f, precision, recall, fscore])
            except IOError as e:
                print('Could not process data.\nError', e, '- Skipping file.')
        # metrics_writer.writerow(['avg', np.mean(precision_ar), np.mean(recall_ar), np.mean(fscore_ar)])
        # metrics_writer.writerow(['std', np.std(precision_ar), np.std(recall_ar), np.std(fscore_ar)])

        return gt_scores, pred_scores

        # fpr_rf, tpr_rf, _ = roc_curve(gt_scores, pred_scores)
        #
        # return fpr_rf, tpr_rf

        # plt.figure(1)
        # plt.plot([0, 1], [0, 1], 'k--')
        # plt.plot(fpr_rf, tpr_rf, label='Fold 0')
        # plt.xlabel('False positive rate')
        # plt.ylabel('True positive rate')
        # plt.title('ROC curve')
        # plt.legend(loc='best')
        # plt.show()


def get_metrics_roc_fold(pred_folder, gt_folder, weights_folder, cv_folder=None, pixel_depth=255):

    if cv_folder is not None:

        auc_ar = np.zeros((5))

        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []

        # plt.figure()
        fig, ax = plt.subplots()

        for fold in range(0,5):
            kk = '/fold' + str(fold) + '/Done'
            cv_folder_fold = cv_folder + '/fold' + str(fold) + '/Done'
            pred_folder_fold = pred_folder + '/pych_cnptool_f' + str(fold) + '/out/out' + str(fold) + '/auto_rev/ch1'
            gt_scores, pred_scores = get_fold_scores(pred_folder_fold, gt_folder, weights_folder, cv_folder_fold)

            fpr_rf, tpr_rf, _ = roc_curve(gt_scores, pred_scores)

            auc_ar[fold] = auc(fpr_rf, tpr_rf)

            plt.plot(fpr_rf, tpr_rf,
                     alpha=0.4,
                     label='Fold ' + str(fold) + ' (AUC=' + str(round(auc_ar[fold], 2)) + ')')

            interp_tpr = np.interp(mean_fpr, fpr_rf, tpr_rf)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc_ar[fold])

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f (%0.2f))' % (mean_auc, std_auc),
                lw=2, alpha=.8)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.5,
                        label=r'$\pm$ 1 std. dev.')

        plt.plot([0, 1], [0, 1], 'k--', label='Chance')
        plt.xlim([0.0, 1.0])#1.02])
        plt.ylim([0.0, 1.0])#1.02])
        plt.gca().set_aspect('equal')#, adjustable='box')
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        # plt.xlabel('1 - Specificity')
        # plt.ylabel('Sensitivity')
        plt.title('ROC curves')
        plt.legend(loc='best')
        plt.show()

        print('auc:', auc_ar)
        print('mean:', np.mean(auc_ar))
        print('std:', np.std(auc_ar))

        ###############################

        auprc_ar = np.zeros((5))

        mean_rec = np.linspace(0, 1, 100)
        tprs = []
        aucs = []

        # plt.figure()
        fig, ax = plt.subplots()
        # plt.plot([1, 0], [0, 1], 'k--')

        for fold in range(0, 5):
            kk = '/fold' + str(fold) + '/Done'
            cv_folder_fold = cv_folder + '/fold' + str(fold) + '/Done'
            pred_folder_fold = pred_folder + '/pych_cnptool_f' + str(fold) + '/out/out' + str(fold) + '/auto_rev/ch1'

            gt_scores, pred_scores = get_fold_scores(pred_folder_fold, gt_folder, weights_folder, cv_folder_fold)

            precision_ar, recall_ar, _ = precision_recall_curve(gt_scores, pred_scores)

            auprc_ar[fold] = auc(recall_ar, precision_ar)

            #
            precision_ar2 = np.append(0, precision_ar)
            recall_ar2 = np.append(1, recall_ar)
            #

            plt.plot(precision_ar2, recall_ar2,
                     alpha=0.4,
                     label='Fold ' + str(fold) + ' (AUPRC=' + str(round(auprc_ar[fold], 2)) + ')')

            interp_prec = np.interp(mean_rec, precision_ar2, recall_ar2)
            interp_prec[0] = 1.0
            tprs.append(interp_prec)
            aucs.append(auprc_ar[fold])

        mean_prec = np.mean(tprs, axis=0)
        mean_prec[-1] = 0.0
        mean_auc = auc(mean_rec, mean_prec)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_prec, color='b',
                label=r'Mean PRC (AUPRC = %0.2f (%0.2f))' % (mean_auc, std_auc),
                lw=2, alpha=.8)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_prec + std_tpr, 1)
        tprs_lower = np.maximum(mean_prec - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.5,
                        label=r'$\pm$ 1 std. dev.')

        plt.plot([1, 0], [0, 1], 'k--', label='Chance')
        plt.xlim([0.0, 1.0])#1.02])
        plt.ylim([0.0, 1.0])#1.02])
        plt.gca().set_aspect('equal')#, adjustable='box')
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PRC curves')
        plt.legend(loc='best')
        plt.show()

        print('auprc:', auprc_ar)
        print('mean:', np.mean(auprc_ar))
        print('std:', np.std(auprc_ar))


    else:
        gt_scores, pred_scores = get_fold_scores(pred_folder, gt_folder, weights_folder)

        fpr_rf, tpr_rf, _ = roc_curve(gt_scores, pred_scores)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curves')
        plt.legend(loc='best')
        plt.show()

        print('auc:', auc_ar)
        print('mean:', np.mean(auc_ar))
        print('std:', np.std(auc_ar))


def get_metrics_roc_fold_BU(pred_folder, gt_folder, weights_folder, cv_folder=None, pixel_depth=255):

    if cv_folder is not None:

        auc_ar = np.zeros((5))

        # plt.figure()
        fig, ax = plt.subplots()

        for fold in range(0,5):
            kk = '/fold' + str(fold) + '/Done'
            cv_folder_fold = cv_folder + '/fold' + str(fold) + '/Done'
            pred_folder_fold = pred_folder + '/pych_cnptool_f' + str(fold) + '/out/out' + str(fold) + '/auto_rev/ch1'
            gt_scores, pred_scores = get_fold_scores(pred_folder_fold, gt_folder, weights_folder, cv_folder_fold)

            fpr_rf, tpr_rf, _ = roc_curve(gt_scores, pred_scores)

            auc_ar[fold] = auc(fpr_rf, tpr_rf)

            plt.plot(fpr_rf, tpr_rf, label='Fold ' + str(fold) + ' (AUC=' + str(round(auc_ar[fold], 2)) + ')')

        plt.plot([0, 1], [0, 1], 'k--', label='Chance')
        plt.xlim([0.0, 1.02])
        plt.ylim([0.0, 1.02])
        plt.gca().set_aspect('equal')#, adjustable='box')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        # plt.xlabel('1 - Specificity')
        # plt.ylabel('Sensitivity')
        plt.title('ROC curves')
        plt.legend(loc='best')
        plt.show()

        print('auc:', auc_ar)
        print('mean:', np.mean(auc_ar))
        print('std:', np.std(auc_ar))

        ###############################

        auprc_ar = np.zeros((5))

        # plt.figure()
        fig, ax = plt.subplots()
        # plt.plot([1, 0], [0, 1], 'k--')

        for fold in range(0, 5):
            kk = '/fold' + str(fold) + '/Done'
            cv_folder_fold = cv_folder + '/fold' + str(fold) + '/Done'
            pred_folder_fold = pred_folder + '/pych_cnptool_f' + str(fold) + '/out/out' + str(fold) + '/auto_rev/ch1'

            gt_scores, pred_scores = get_fold_scores(pred_folder_fold, gt_folder, weights_folder, cv_folder_fold)

            precision_ar, recall_ar, _ = precision_recall_curve(gt_scores, pred_scores)

            auprc_ar[fold] = auc(recall_ar, precision_ar)

            #
            precision_ar2 = np.append(0, precision_ar)
            recall_ar2 = np.append(1, recall_ar)
            #

            plt.plot(precision_ar2, recall_ar2, label='Fold ' + str(fold) + ' (AUPRC=' + str(round(auprc_ar[fold], 2)) + ')')

        plt.plot([1, 0], [0, 1], 'k--', label='Chance')
        plt.xlim([0.0, 1.02])
        plt.ylim([0.0, 1.02])
        plt.gca().set_aspect('equal')#, adjustable='box')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curves')
        plt.legend(loc='best')
        plt.show()

        print('auprc:', auprc_ar)
        print('mean:', np.mean(auprc_ar))
        print('std:', np.std(auprc_ar))


    else:
        gt_scores, pred_scores = get_fold_scores(pred_folder, gt_folder, weights_folder)

        fpr_rf, tpr_rf, _ = roc_curve(gt_scores, pred_scores)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curves')
        plt.legend(loc='best')
        plt.show()

        print('auc:', auc_ar)
        print('mean:', np.mean(auc_ar))
        print('std:', np.std(auc_ar))