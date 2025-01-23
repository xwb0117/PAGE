import torch
from torch.autograd import Variable
from torch import autograd
from tensorboardX import SummaryWriter

def cal_loss_dispersion(input, gamma, delta_max):
    '''
    input: keypoint tensor [b,n,3]
            b-batch
            n-number of points
            3-x,y,z

    output: dispersion loss
    '''
    n = input.size()[1]
    iter = 0
    sum = 0
    for kpoints in input:
        for i in range(0, n - 1):
            for j in range(i + 1, n):
                # if i!=j:
                # print("[",i,", ",j,"]")
                distance = ((kpoints[i, 0] - kpoints[j, 0]) ** 2 + (kpoints[i, 1] - kpoints[j, 1]) ** 2 + (
                            kpoints[i, 2] - kpoints[j, 2]) ** 2) ** 0.5
                # Apply the exponential penalty for large distances
                # sum += torch.exp(gamma * distance)
                sum += torch.max(torch.tensor(0), delta_max - distance)
                iter += 1
    return sum / iter

def smooth_l1_loss(pred, target, beta=1.0, reduction='mean'):
    assert reduction in ('mean', 'none')
    assert beta > 0
    assert pred.size() == target.size()
    if target.numel() == 0:
        return pred * 0.
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'none':
        return loss

def CoverageLoss(kp, pc, beta=1.0, gamma=torch.tensor(2), use_relative_coverage=False):
    if use_relative_coverage:
        # volume
        val_max_pc, _ = torch.max(pc, 2)
        val_min_pc, _ = torch.min(pc, 2)

        val_max_kp, _ = torch.max(kp, 2)
        val_min_kp, _ = torch.min(kp, 2)

        scale_pc = val_max_pc - val_min_pc
        scale_kp = val_max_kp - val_min_kp

        cov_loss = (smooth_l1_loss(val_max_kp / scale_pc, val_max_pc / scale_pc, beta=beta) +
                    smooth_l1_loss(val_min_kp / scale_pc, val_min_pc / scale_pc, beta=beta) +
                    smooth_l1_loss(torch.log(scale_kp), torch.log(scale_pc), beta=beta)
                    ) / 3
    else:
        # volume
        val_max_pc, _ = torch.max(pc, 2)
        val_min_pc, _ = torch.min(pc, 2)

        val_max_pc = val_max_pc * gamma
        val_min_pc = val_max_pc * gamma

        val_max_kp, _ = torch.max(kp, 2)
        val_min_kp, _ = torch.min(kp, 2)

        cov_loss = (smooth_l1_loss(val_max_kp, val_max_pc, beta=beta) +
                    smooth_l1_loss(val_min_kp, val_min_pc, beta=beta))/2
    return cov_loss

def cal_loss_wass_gp(input, segments, vote_type, model, device, lambda_, training):
    '''
    input: keypoint tensor [b,n,3]
            b-batch
            n-number of points
            3-x,y,z
    segment: segment point cloud tensor [b,N,3]
            b-batch
            N-number of points
            3-x,y,z
    return wass loss
    '''
    b = input.size()[0]  #batch_size
    n = input.size()[1]
    N = segments.size()[2]
    sum = 0
    iter = 0
    for i in range(b):
        kpoints = input[i]
        segment = segments[i]
        segment = segment.permute(1, 0)
        votes = torch.zeros(n, N, 3).cuda(device)
        norms = torch.zeros(n, N).cuda(device)
        sorted_norms = torch.zeros(n, N).cuda(device)
        sorted_indices = torch.zeros(n, N, dtype=torch.long).cuda(device)
        for j in range(n):
            kpoint = kpoints[j]
            un_squeezed_kpt = torch.unsqueeze(kpoint, 0)
            un_squeezed_kpt = un_squeezed_kpt.repeat(N, 1)
            xyz = segment[:, :3]
            offsets = torch.sub(un_squeezed_kpt, xyz)
            norm = torch.norm(offsets, dim=1)
            norms[j] = norm
            sorted_norms[j], sorted_indices[j] = torch.sort(norms[j])

        for j in range(0, n - 1):
            for k in range(j + 1, n):
                wass_dist = torch.mean(torch.abs(sorted_norms[j] - sorted_norms[k]))
                sum += wass_dist
                iter += 1
    l_wass = sum / iter
    if training:
        idx = np.random.choice(np.arange(b), b, replace=False)
        shuffled = segments[idx, :, :]
        shuffled = Variable(shuffled, requires_grad=True)

        prob_shuffled = model(shuffled)
        gradients = autograd.grad(outputs=prob_shuffled, inputs=shuffled,
                                  grad_outputs=torch.ones(
                                      prob_shuffled.size()).cuda(device) if args.cuda else torch.ones(
                                      prob_shuffled.size()),
                                  create_graph=True, retain_graph=True)[0]
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_
        l_wass += grad_penalty
    return l_wass

if __name__ == "__main__":
    import os
    import argparse
    import tqdm
    import shutil
    import numpy as np
    from models.KPNet import KP_Estimator
    from dataset.dataloader_kp import ArtDataset

    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser(description='KPNet Test')
    parser.add_argument('--keypointsNo', type=int, default=3, metavar='keypointsNo',
                        help='No, of Keypoints')

    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')

    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of test batch)')

    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of epochs to train ')

    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'],
                        help='optimizer for training')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1e-3)')

    parser.add_argument('--cuda', type=bool, default=True,
                        help='enables GPU training')

    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points forwarded into the network')

    parser.add_argument('--min_vis_points', type=int, default=2000,
                        help='threshold for minimum segment points')

    parser.add_argument('--data_root', type=str, default='data/lm',
                        help='dataset root dir')

    parser.add_argument('--ckpt_root', type=str, default='',
                        help='root dir for ckpt')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dgcnn dropout rate')

    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='dgcnn dimension of embeddings')

    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='dgcnn random seed (default: 1)')

    parser.add_argument('--lambda_term', type=float, default=1,
                        help='gradient penalty lambda term')

    parser.add_argument('--gamma', type=float, default=-0.5,
                        help='gamma for dispersion loss')

    parser.add_argument('--vote_type', type=int, default=0, choices=[0, 1, 2],
                        help='vote type to train on. radii:0, vector:1, offset:2.')
    parser.add_argument('--category', type=str, default='laptop',
                        help='category of articulated objects')
    parser.add_argument('--nparts', type=int, default=2,
                        help='parts nums of articulated objects')
    parser.add_argument('--part_num', type=int, default=0,
                        help='parts num of articulated objects')
    parser.add_argument('--dname', type=str, default='Art',
                        help='dataset name')
    parser.add_argument('--scale', type=int, default=1,
                        help='scale factor')
    vote_types = {0: 'radii'}
    args = parser.parse_args()
    dataset_name = args.data_root.split('/')[-1]
    log_dir = os.path.join('logs_kp', dataset_name, vote_types[args.vote_type], f'{args.category}_{args.part_num}_{args.keypointsNo}')
    print(log_dir)
    # create log root
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if args.cuda and torch.cuda.is_available():
        print("Using GPU!")
        device = 'cuda:0'
        torch.cuda.manual_seed(args.seed)
    else:
        device = 'cpu'
        torch.manual_seed(args.seed)

    # initialize model, optimizer, and dataloader; load ckpt
    model = KP_Estimator(args, device).to(device)
    if args.optim == 'Adam':
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    if args.ckpt_root != '':
        if os.path.isfile(args.ckpt_root):
            checkpoint = torch.load(args.ckpt_root)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.ckpt_root))

    tblogger = SummaryWriter(logdir=os.path.join(log_dir, 'tblog'))

    model.train()
    train_loader = torch.utils.data.DataLoader(ArtDataset(args.data_root, 'train',
                                                          min_visible_points=args.min_vis_points,
                                                          points_count_net=args.num_points,
                                                          cat=args.category,
                                                          n_parts=args.nparts,
                                                          part_num=args.part_num,
                                                          dname=args.dname,
                                                          scale=args.scale),
                                               batch_size=args.batch_size, shuffle=False,num_workers=6)
    test_loader = torch.utils.data.DataLoader(ArtDataset(args.data_root, 'test',
                                                         min_visible_points=args.min_vis_points,
                                                         points_count_net=args.num_points,
                                                         cat=args.category,
                                                         n_parts=args.nparts,
                                                         part_num=args.part_num,
                                                         dname=args.dname,
                                                         scale=args.scale),
                                              batch_size=args.test_batch_size, shuffle=False,num_workers=6)
    train_iteration = 0
    test_iteration = 0
    best_test_loss = np.inf
    save_name = "ckpt.pth.tar"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, verbose=True)
    for epoch in range(args.epochs):
        for batch_idx, pc in tqdm.tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc='Train epoch=%d' % epoch,
                ncols=80,
                leave=False):
            pc = torch.permute(pc, (0, 2, 1))
            pc = pc.to(device)
            training = True
            optim.zero_grad()

            est_kpts = model(pc)

            loss_dis = cal_loss_dispersion(est_kpts, args.gamma, torch.tensor(1).to(device))
            # print(loss_dis)
            loss_wass = cal_loss_wass_gp(est_kpts, pc, args.vote_type, model, device, args.lambda_term, training)

            loss_cov = CoverageLoss(est_kpts,pc)
            # loss_cov = torch.tensor(0.).to(device)

            alpha = 10
            beta = 1

            loss = alpha * loss_wass + beta * loss_dis + loss_cov
            loss.backward()
            optim.step()
            np_loss_dis, np_loss_wass, np_loss_cov, np_loss = loss_dis.detach().cpu().numpy(), loss_wass.detach().cpu().numpy(), loss_cov.detach().cpu().numpy(),loss.detach().cpu().numpy()
            tblogger.add_scalar('Train', np_loss, train_iteration)
            tblogger.add_scalar('Train_wass', np_loss_wass, train_iteration)
            tblogger.add_scalar('Train_disp', np_loss_dis, train_iteration)
            tblogger.add_scalar('Train_cov', np_loss_cov, train_iteration)
            print('Train', np_loss, train_iteration)
            print('Train_wass', np_loss_wass, train_iteration)
            print('Train_disp', np_loss_dis, train_iteration)
            print('Train_cov', np_loss_cov, train_iteration)
            train_iteration += 1

        test_losses = []
        # dump ckpt

        torch.save(
            {
                'epoch': epoch,
                'optim_state_dict': optim.state_dict(),
                'model_state_dict': model.state_dict(),
            }, os.path.join(log_dir, save_name))

        # test loop
        if epoch % 5 == 0 and epoch != 0:
            with torch.no_grad():
                for batch_idx, pc in tqdm.tqdm(
                        enumerate(test_loader),
                        total=len(test_loader),
                        desc='Test for epoch %d' % epoch,
                        ncols=80,
                        leave=False):
                    pc = torch.permute(pc, (0, 2, 1))
                    pc = pc.to(device)
                    est_kpts = model(pc)
                    loss_dis = cal_loss_dispersion(est_kpts, args.gamma, torch.tensor(1).to(device))
                    training = False
                    loss_wass = cal_loss_wass_gp(est_kpts, pc, args.vote_type, model, device, args.lambda_term,
                                                 training)
                    loss_cov = CoverageLoss(est_kpts, pc)
                    loss = alpha * loss_wass + beta * loss_dis
                    np_loss_dis, np_loss_wass, np_loss_cov, np_loss = loss_dis.detach().cpu().numpy(), loss_wass.detach().cpu().numpy(), loss_cov.detach().cpu().numpy(), loss.detach().cpu().numpy()
                    tblogger.add_scalar('Test', np_loss, train_iteration)
                    tblogger.add_scalar('Test_wass', np_loss_wass, train_iteration)
                    tblogger.add_scalar('Test_disp', np_loss_dis, train_iteration)
                    tblogger.add_scalar('Test_cov', np_loss_cov, train_iteration)
                    print('Test', np_loss, train_iteration)
                    print('Test_wass', np_loss_wass, train_iteration)
                    print('Test_disp', np_loss_dis, train_iteration)
                    print('Test_cov', np_loss_cov, train_iteration)
                    test_iteration += 1
                    test_losses.append(np_loss)
                test_loss = np.mean(np.array(test_losses))
                scheduler.step(test_loss)

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    # replace best model
                    shutil.copy(os.path.join(log_dir, save_name),
                                os.path.join(log_dir, 'model_best.pth.tar'))