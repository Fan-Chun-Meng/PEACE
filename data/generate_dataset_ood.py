'''
 every node have different observations
        train observation length [ob_min, ob_max]
'''

from synthetic_sim import ChargedParticlesSim, SpringSim
import time
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--simulation', type=str, default='springs',
                    help='What simulation to generate.')
parser.add_argument('--num-train', type=int, default=1000,
                    help='Number of training simulations to generate.')
parser.add_argument('--num-test', type=int, default=200,
                    help='Number of test simulations to generate.')
parser.add_argument('--num-val', type=int, default=200,
                    help='Number of valid simulations to generate.')
parser.add_argument('--ode', type=int, default=2400,
                    help='Length of trajectory.')
parser.add_argument('--num-test-box', type=int, default=1,
                    help='Length of test set trajectory.')
parser.add_argument('--num-test-extra', type=int, default=24,
                    help='Length of test set trajectory.')
parser.add_argument('--sample-freq', type=int, default=100,
                    help='Length of test set trajectory.')
parser.add_argument('--ob_max', type=int, default=24,
                    help='Length of test set trajectory.')
parser.add_argument('--ob_min', type=int, default=24,
                    help='Length of test set trajectory.')
parser.add_argument('--n-balls', type=int, default=10,
                    help='Number of balls in the simulation.')
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed.')

args = parser.parse_args()

if args.simulation == 'springs':
    # sim = SpringSim(noise_var=0.0, n_balls=args.n_balls)
    suffix = "_springs"

elif args.simulation == 'charged':
    # sim = ChargedParticlesSim(noise_var=0.0, n_balls=args.n_balls)
    suffix = '_charged'
else:
    raise ValueError('Simulation {} not implemented'.format(args.simulation))

suffix += str(args.n_balls)
np.random.seed(args.seed)

print(suffix)


def create_mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)


def create_ood_param_test(num_points, lows, highs, lows_out, highs_out):
    a_low, b_low, c_low, d_low = lows
    a_high, b_high, c_high, d_high = highs
    a_low_out, b_low_out, c_low_out, d_low_out = lows_out
    a_high_out, b_high_out, c_high_out, d_high_out = highs_out

    params = np.zeros((num_points, 4))
    i = 0
    while (i < num_points):
        a = np.random.uniform(a_low_out, a_high_out)
        b = np.random.uniform(b_low_out, b_high_out)
        c = np.random.uniform(c_low_out, c_high_out)
        d = np.random.uniform(d_low_out, d_high_out)

        if not (
                ((a >= a_low) and (a <= a_high))
                and ((b >= b_low) and (b <= b_high))
                and ((c >= c_low) and (c <= c_high))
                and ((d >= d_low) and (d <= d_high))
        ):
            params[i] = np.array([a, b, c, d])
            i += 1
        else:
            pass
    #             print(np.round([a,b,c,d],2))
    return params


def generate_dataset(args,num_sims,isTrain = True):
    box_size_min = 4.9
    box_size_max = 5.1
    vel_norm_min = 0.49
    vel_norm_max = 0.51
    interaction_strength_min = 0.09
    interaction_strength_max = 0.11
    spring_prob_min = 0.49
    spring_prob_max = 0.51

    box_size_min_out = 4.8
    box_size_max_out = 5.2
    vel_norm_min_out = 0.48
    vel_norm_max_out = 0.52
    interaction_strength_min_out = 0.08
    interaction_strength_max_out = 0.12
    spring_prob_min_out = 0.48
    spring_prob_max_out = 0.52

    lows = [box_size_min, vel_norm_min, interaction_strength_min, spring_prob_min]
    highs = [box_size_max, vel_norm_max, interaction_strength_max, spring_prob_max]
    lows_out = [box_size_min_out, vel_norm_min_out, interaction_strength_min_out, spring_prob_min_out]
    highs_out = [box_size_max_out, vel_norm_max_out, interaction_strength_max_out, spring_prob_max_out]
    paras_ood = create_ood_param_test(num_sims, lows=lows, highs=highs, lows_out=lows_out, highs_out=highs_out)

    loc_all = list()
    vel_all = list()
    edges = list()
    timestamps = list()
    sys_paras = list()

    for i in range(num_sims):

        box_size, vel_norm, interaction_strength, spring_prob_value = paras_ood[i]
        sys_paras.append([box_size, vel_norm, interaction_strength, spring_prob_value])

        sim = SpringSim(noise_var=0.0, n_balls=args.n_balls, box_size=box_size, vel_norm=vel_norm,
                        interaction_strength=interaction_strength)

        t = time.time()
        #graph generation
        static_graph = sim.generate_static_graph(spring_prob=[1-spring_prob_value, 0, spring_prob_value])
        edges.append(static_graph)  # [5,5]

        loc, vel, T_samples = sim.sample_trajectory_static_graph_v2(args, edges=static_graph, isTrain=isTrain)
        # print(123)
        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
        loc_all.append(loc)  # [5,]
        vel_all.append(vel)  # [49,2,5]
        timestamps.append(T_samples)  # [99]

    loc_all = np.asarray(loc_all)  # [5000,5 list(timestamps,2)]
    vel_all = np.asarray(vel_all)
    edges = np.stack(edges)
    timestamps = np.asarray(timestamps)
    sys_paras = np.asarray(sys_paras)

    return loc_all, vel_all, edges, timestamps, sys_paras


def generate_dataset_charged(args,num_sims,isTrain = True):
    box_size_min = 4.9
    box_size_max = 5.1
    vel_norm_min = 0.49
    vel_norm_max = 0.51
    interaction_strength_min = 0.9
    interaction_strength_max = 1.1
    charge_prob_min = 0.49
    charge_prob_max = 0.51

    box_size_min_out = 4.8
    box_size_max_out = 5.2
    vel_norm_min_out = 0.48
    vel_norm_max_out = 0.52
    interaction_strength_min_out = 0.8
    interaction_strength_max_out = 1.2
    charge_prob_min_out = 0.48
    charge_prob_max_out = 0.52


    lows = [box_size_min, vel_norm_min, interaction_strength_min, charge_prob_min]
    highs = [box_size_max, vel_norm_max, interaction_strength_max, charge_prob_max]
    lows_out = [box_size_min_out, vel_norm_min_out, interaction_strength_min_out, charge_prob_min_out]
    highs_out = [box_size_max_out, vel_norm_max_out, interaction_strength_max_out, charge_prob_max_out]
    paras_ood = create_ood_param_test(num_sims, lows=lows, highs=highs, lows_out=lows_out, highs_out=highs_out)

    loc_all = list()
    vel_all = list()
    edges = list()
    timestamps = list()
    sys_paras = list()

    for i in range(num_sims):

        box_size, vel_norm, interaction_strength, charge_prob_value = paras_ood[i]
        sys_paras.append([box_size, vel_norm, interaction_strength, charge_prob_value])

        sim = ChargedParticlesSim(noise_var=0.0, n_balls=args.n_balls, box_size=box_size, vel_norm=vel_norm,
                        interaction_strength=interaction_strength)

        t = time.time()
        #graph generation
        static_graph,diag_mask = sim.generate_static_graph(charge_prob=[1-charge_prob_value, 0, charge_prob_value])
        edges.append(static_graph)  # [5,5]

        loc, vel, T_samples = sim.sample_trajectory_static_graph_v2(args, edges=static_graph,diag_mask = diag_mask,
                                                                                               isTrain=isTrain)
        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
        loc_all.append(loc)  # [49,2,5]
        vel_all.append(vel)  # [49,2,5]
        timestamps.append(T_samples)  # [99]

    loc_all = np.asarray(loc_all)  # [5000,5 list(timestamps,2)]
    vel_all = np.asarray(vel_all)
    edges = np.stack(edges)
    timestamps = np.asarray(timestamps)
    sys_paras = np.asarray(sys_paras)

    return loc_all, vel_all, edges, timestamps, sys_paras


if args.simulation =="springs":
    dir_name = args.simulation+'_{}_{}_{}_ood'.format(args.num_train, args.n_balls, args.num_test_extra)
    create_mkdir('{}/'.format(dir_name))
    print("Generating {} test simulations".format(args.num_test))

    loc_test, vel_test, edges_test, timestamps_test, paras_test = generate_dataset(args, args.num_test, isTrain=False)
    np.save(dir_name+'/loc_test' + suffix + '.npy', loc_test)
    np.save(dir_name+'/vel_test' + suffix + '.npy', vel_test)
    np.save(dir_name+'/edges_test' + suffix + '.npy', edges_test)
    np.save(dir_name+'/times_test' + suffix + '.npy', timestamps_test)
    np.save(dir_name + '/paras_test' + suffix + '.npy', paras_test)

    # print("Generating {} valid simulations".format(args.num_test))
    # loc_val, vel_val, edges_val, timestamps_val, paras_val = generate_dataset(args, args.num_val, isTrain=False)
    # np.save(dir_name+'/loc_val' + suffix + '.npy', loc_val)
    # np.save(dir_name+'/vel_val' + suffix + '.npy', vel_val)
    # np.save(dir_name+'/edges_val' + suffix + '.npy', edges_val)
    # np.save(dir_name+'/times_val' + suffix + '.npy', timestamps_val)
    # np.save(dir_name + '/paras_val' + suffix + '.npy', paras_val)
    #
    # print("Generating {} training simulations".format(args.num_train))
    # loc_train, vel_train, edges_train, timestamps_train, paras_train = generate_dataset(args, args.num_train, isTrain=True)
    #
    # np.save(dir_name+'/loc_train' + suffix + '.npy', loc_train)
    # np.save(dir_name+'/vel_train' + suffix + '.npy', vel_train)
    # np.save(dir_name+'/edges_train' + suffix + '.npy', edges_train)
    # np.save(dir_name+'/times_train' + suffix + '.npy', timestamps_train)
    # np.save(dir_name + '/paras_train' + suffix + '.npy', paras_train)


elif args.simulation == "charged":
    dir_name = args.simulation + '_{}_{}_{}_ood/'.format(args.num_train, args.n_balls, args.num_test_extra)
    create_mkdir('{}'.format(dir_name))

    print("Generating {} test simulations".format(args.num_test))

    loc_test, vel_test, edges_test, timestamps_test, paras_test = generate_dataset_charged(args, args.num_test, isTrain=False)
    np.save(dir_name+'loc_test' + suffix + '.npy', loc_test)
    np.save(dir_name+'vel_test' + suffix + '.npy', vel_test)
    np.save(dir_name+'edges_test' + suffix + '.npy', edges_test)
    np.save(dir_name+'times_test' + suffix + '.npy', timestamps_test)
    np.save(dir_name + 'paras_test' + suffix + '.npy', paras_test)

    # print("Generating {} valid simulations".format(args.num_val))
    # loc_val, vel_val, edges_val, timestamps_val, paras_val = generate_dataset_charged(args, args.num_val, isTrain=False)
    # np.save(dir_name+'loc_val' + suffix + '.npy', loc_val)
    # np.save(dir_name+'vel_val' + suffix + '.npy', vel_val)
    # np.save(dir_name+'edges_val' + suffix + '.npy', edges_val)
    # np.save(dir_name+'times_val' + suffix + '.npy', timestamps_val)
    # np.save(dir_name + 'paras_val' + suffix + '.npy', paras_val)
    #
    # print("Generating {} training simulations".format(args.num_train))
    # loc_train, vel_train, edges_train, timestamps_train, paras_train = generate_dataset_charged(args, args.num_train, isTrain=True)
    #
    # np.save(dir_name+'loc_train' + suffix + '.npy', loc_train)
    # np.save(dir_name+'vel_train' + suffix + '.npy', vel_train)
    # np.save(dir_name+'edges_train' + suffix + '.npy', edges_train)
    # np.save(dir_name+'times_train' + suffix + '.npy', timestamps_train)
    # np.save(dir_name + 'paras_train' + suffix + '.npy', paras_train)




