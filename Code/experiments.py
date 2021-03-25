from data import get_data, get_data_source
from bandits.storage import (Action, MemoryHistoryStorage, MemoryModelStorage, MemoryActionStorage) 
from bandits.storage import history
from bandits.storage import model
from bandits.bandit import (linucb, linthompsamp)
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from selection import scheme_selection
import logging
from pathlib import Path
from numpy.random import Generator, PCG64


def run_bandit_arm(dt):
    n_rounds = 1000
    #tot_arms = 100
    candidate_ix = [10, 100, 250, 500]

    df, X, farms, anchor_ids, anchor_features, tot_arms = get_data(dt)
    #rg = Generator(PCG64(12345))
    #anchor_ids = rg.choice(anchor_ids,tot_arms,replace=False)
    bandit = 'LinUCB'
    scheme = 'submodular'  #scheme = 'random'
    src = get_data_source(dt)
    regret = {}
    epsilon = 0.5
    for cand_sz in candidate_ix:
        regret[cand_sz] = {}
        log_file = Path('../Data/', src, src+'_%d.log' %(cand_sz))
        logging.basicConfig(filename = log_file, format='%(asctime)s : %(message)s', level=logging.INFO)
        logging.info("Running %s algorithm with %s selection scheme for epsilon %f with candidate size %d" %(bandit,scheme,epsilon,cand_sz))
        for anchor in anchor_ids:
            anchor_session_id = df.iloc[anchor]['session_id']
            true_ids = df.index[df['session_id'] == anchor_session_id].tolist()
            logging.info("Calculating cosine similarity")
            cos_sim = cosine_similarity(X[anchor,:].reshape(1,-1),X)
            logging.info("Running arms selection algorithm for anchor id: %d session_id: %s" %(anchor, anchor_session_id))
            actions = scheme_selection(scheme,farms[farms != anchor],np.delete(cos_sim.ravel(),anchor),cand_sz,epsilon)
            logging.info("Finished with arms selection")
            arms_context = X[actions,:]
            anchor_context = X[anchor,:]    #.reshape(1,-1)
            arms = MemoryActionStorage()
            arms.add([Action(act) for act in actions])
            regret[cand_sz][anchor] = {}
            policy = policy_generation(bandit, arms)
            logging.info("evaluating policy")
            seq_error = policy_evaluation(policy, bandit, anchor, anchor_context, true_ids, arms, arms_context,n_rounds)
            logging.info("calculating regret")
            regret[cand_sz][anchor] = regret_calculation(seq_error)
            logging.info("finished with regret calculation")

        logger = logging.getLogger()
        for hdlr in logger.handlers[:]:
            hdlr.close()
            logger.removeHandler(hdlr)

    import matplotlib.pyplot as plt
    from matplotlib import rc
    f = plt.figure()
    f.clear()
    plt.clf()
    plt.close(f)
    with plt.style.context(("seaborn-darkgrid",)):
        fig, ax = plt.subplots(frameon=False)
        rc('mathtext',default='regular')
        rc('text', usetex=True)
        col = {10:'b', 100:'r', 250:'k', 500:'c'}
        regret_file = 'cand_cum_regret.txt'
        with open(regret_file, "w") as regret_fd:
            for cand_sz in candidate_ix:
                cum_regret = [sum(x)/tot_arms for x in zip(*regret[cand_sz].values())]
                val = str(cand_sz)+','+','.join([str(e) for e in cum_regret])
                print(val, file=regret_fd)
                ax.plot(range(n_rounds), cum_regret, c=col[cand_sz], ls='-', label=r'$k = {}$'.format(cand_sz))
                ax.set_xlabel(r'k')
                ax.set_ylabel(r'cumulative regret')
                ax.legend()
            fig.savefig('arm_regret.pdf',format='pdf')
            f = plt.figure()
            f.clear()
            plt.close(f)


def run_bandit_sim(dt):
    candidate_set_sz = 250
    n_rounds = 1000
    #tot_arms = 100
    epsilon_ix = [0.2, 0.4, 0.6, 0.9]

    df, X, farms, anchor_ids, anchor_features, tot_arms = get_data(dt)
    #rg = Generator(PCG64(12345))
    #anchor_ids = rg.choice(anchor_ids,tot_arms,replace=False)
    bandit = 'LinUCB'
    scheme = 'submodular'   #scheme = 'random'
    src = get_data_source(dt)
    regret = {}
    for psilon in epsilon_ix:
        regret[psilon] = {}
        log_file = Path('../Data/', src, src+'_%f.log' %(psilon))
        logging.basicConfig(filename = log_file, format='%(asctime)s : %(message)s', level=logging.INFO)
        logging.info("Running %s algorithm with %s selection scheme for epsilon %f" %(bandit,scheme,psilon))
        for anchor in anchor_ids:
            anchor_session_id = df.iloc[anchor]['session_id']
            true_ids = df.index[df['session_id'] == anchor_session_id].tolist()
            logging.info("Calculating cosine similarity")
            cos_sim = cosine_similarity(X[anchor,:].reshape(1,-1),X)
            logging.info("Running arms selection algorithm for anchor id: %d session_id: %s" %(anchor, anchor_session_id))
            actions = scheme_selection(scheme,farms[farms != anchor],np.delete(cos_sim.ravel(),anchor),candidate_set_sz,psilon)
            logging.info("Finished with arms selection")
            arms_context = X[actions,:]
            anchor_context = X[anchor,:]    #.reshape(1,-1)
            arms = MemoryActionStorage()
            arms.add([Action(act) for act in actions])
            regret[psilon][anchor] = {}
            policy = policy_generation(bandit, arms)
            logging.info("evaluating policy")
            seq_error = policy_evaluation(policy, bandit, anchor, anchor_context, true_ids, arms, arms_context,n_rounds)
            logging.info("calculating regret")
            regret[psilon][anchor] = regret_calculation(seq_error)
            logging.info("finished with regret calculation")

        logger = logging.getLogger()
        for hdlr in logger.handlers[:]:
            hdlr.close()
            logger.removeHandler(hdlr)

    import matplotlib.pyplot as plt
    from matplotlib import rc
    with plt.style.context(("seaborn-darkgrid",)):
        f = plt.figure()
        f.clear()
        plt.clf()
        plt.close(f)
        fig, ax = plt.subplots(frameon=False)
        rc('mathtext',default='regular')
        rc('text', usetex=True)
        col = {0.2:'b', 0.4:'r', 0.6:'k', 0.9:'c'}
        regret_file = 'sim_cum_regret.txt'
        with open(regret_file, "w") as regret_fd:
            for psilon in epsilon_ix:
                cum_regret = [sum(x)/tot_arms for x in zip(*regret[psilon].values())]
                val = str(psilon)+','+','.join([str(e) for e in cum_regret])
                print(val, file=regret_fd)
                ax.plot(range(n_rounds), cum_regret, c=col[psilon], ls='-', label=r'$\varepsilon = {}$'.format(psilon))
                ax.set_xlabel(r'$\varepsilon$')
                ax.set_ylabel('cumulative regret')
                ax.legend()
            fig.savefig('sim_regret.pdf',format='pdf')
            f = plt.figure()
            f.clear()
            plt.close(f)


def run_bandit_round(dt):
    candidate_set_sz = 250
    n_rounds = 1000
    #tot_arms = 100
    epsilon = 0.5
     
    df, X, farms, anchor_ids, anchor_features, tot_arms = get_data(dt)
    #rg = Generator(PCG64(12345))
    #anchor_ids = rg.choice(anchor_ids, tot_arms, replace=False)
    experiment_bandit = ['LinThompSamp', 'LinUCB', 'Random', 'Similar']
    selection_scheme = ['random', 'submodular']    #selection_scheme = ['submodular']
    regret = {}
    src = get_data_source(dt)
    for scheme in selection_scheme:
        log_file = Path('../Data/', src, src+'_%s.log' %(scheme))
        logging.basicConfig(filename = log_file, format='%(asctime)s : %(message)s', level=logging.INFO)
        logging.info("Running %s selection scheme" %(scheme))
        regret[scheme] = {}
        for anchor in anchor_ids:
            anchor_session_id = df.iloc[anchor]['session_id']
            true_ids = df.index[df['session_id'] == anchor_session_id].tolist()
            cos_sim = cosine_similarity(X[anchor,:].reshape(1,-1),X)
            logging.info("Running arms selection algorithm for anchor id : %d" %(anchor))
            actions = scheme_selection(scheme,farms[farms != anchor],np.delete(cos_sim.ravel(),anchor),candidate_set_sz,epsilon)
            arms_context = X[actions,:]
            anchor_context = X[anchor,:]    #.reshape(1,-1)
            arms = MemoryActionStorage()
            arms.add([Action(act) for act in actions])
            for bandit in experiment_bandit:
                logging.info("Running %s algorithm with %s selection strategy for session id %s" %(bandit,scheme,anchor_session_id))
                if bandit not in regret[scheme]:
                    regret[scheme][bandit] = {}
                policy = policy_generation(bandit, arms)
                logging.info("evaluating policy")
                seq_error = policy_evaluation(policy, bandit, anchor, anchor_context, true_ids, arms, arms_context,n_rounds)
                logging.info("calculating regret")
                regret[scheme][bandit][anchor] = regret_calculation(seq_error)
                logging.info("finished with regret calculation")
        logger = logging.getLogger()
        for hdlr in logger.handlers[:]:
            hdlr.close()
            logger.removeHandler(hdlr)

    import matplotlib.pyplot as plt
    from matplotlib import rc
    with plt.style.context(("seaborn-darkgrid",)):
        f = plt.figure()
        f.clear()
        plt.clf()
        plt.close(f)
        fig, ax = plt.subplots(frameon=False)
        rc('mathtext',default='regular')
        rc('text', usetex=True)
        col = {'LinUCB':'b', 'LinThompSamp':'r', 'Random':'k', 'Similar':'c'}
        sty = {'submodular':'-', 'random':':'}
        labels = {'LinThompSamp':{'submodular':'LinThompSamp (max-utility)','random':'LinThompSamp (random)'}, 'LinUCB':{'submodular':'LinUCB (max-utility)', 'random':'LinUCB (random)'}, 'Random':{'submodular': 'Random (max-utility)', 'random': 'Random (random)'}, 'Similar':{'submodular': 'Similar (max-utility)', 'random': 'Similar (random)'}}
        regret_file = 'cum_regret.txt'
        with open(regret_file, "w") as regret_fd:
            for scheme in selection_scheme:
                for bandit in experiment_bandit:
                    cum_regret = [sum(x)/tot_arms for x in zip(*regret[scheme][bandit].values())]
                    val = bandit+','+scheme+','+','.join([str(e) for e in cum_regret])
                    print(val, file=regret_fd)
                    ax.plot(range(n_rounds), cum_regret, c=col[bandit], ls=sty[scheme], label=labels[bandit][scheme])
                    ax.set_xlabel('rounds')
                    ax.set_ylabel('cumulative regret')
                    ax.legend()
            fig.savefig('round_regret.pdf',format='pdf')
            f = plt.figure()
            f.clear()
            plt.close(f)


def regret_calculation(seq_error):
    t = len(seq_error)
    regret = [x / y for x, y in zip(seq_error, range(1, t + 1))]
    return regret 

def policy_evaluation(policy, bandit, anchor, anchor_context, true_ids, arms, arms_context,n_rounds):
    seq_error = np.zeros(shape=(n_rounds, 1))
    actions_id = [a for a in arms.iterids()]
    if bandit in ['LinUCB', 'LinThompSamp']:
        for t in range(n_rounds):
            full_context = {}
            for action_id in actions_id:
                #full_context[action_id] = anchor_context
                full_context[action_id] = arms_context[actions_id.index(action_id),:]
            history_id, action = policy.get_action(full_context, n_actions=1)
            if action[0].action.id not in true_ids:
                policy.reward(history_id, {action[0].action.id: 0.0})
                if t == 0:
                    seq_error[t] = 1.0
                else:
                    seq_error[t] = seq_error[t - 1] + 1.0
            else:
                policy.reward(history_id, {action[0].action.id: 1.0})
                if t > 0:
                    seq_error[t] = seq_error[t - 1]
    elif bandit == 'Random':
        rg = Generator(PCG64(12345))
        for t in range(n_rounds):
            action = actions_id[rg.integers(low=0, high=len(actions_id)-1)]
            if action not in true_ids:
                if t == 0:
                    seq_error[t] = 1.0
                else:
                    seq_error[t] = seq_error[t - 1] + 1.0
            else:
                if t > 0:
                    seq_error[t] = seq_error[t - 1]
    elif bandit == 'Similar':
        rg = Generator(PCG64(12345))
        for t in range(n_rounds):
            #anchor_contextselect one of the 5 most similar arm
            cos_sim = cosine_similarity(anchor_context.reshape(1,-1),arms_context)
            ind = np.argpartition(cos_sim.ravel(),-10)[-10:]
            action = actions_id[rg.choice(ind)]
            if action not in true_ids:
                if t == 0:
                    seq_error[t] = 1.0
                else:
                    seq_error[t] = seq_error[t - 1] + 1.0
            else:
                if t > 0:
                    seq_error[t] = seq_error[t - 1]

            
    return seq_error
            

def policy_generation(bandit, arms):
    historystorage = history.MemoryHistoryStorage()
    modelstorage = model.MemoryModelStorage()

    if bandit == 'LinUCB':
        policy = linucb.LinUCB(history_storage=historystorage, model_storage=modelstorage, action_storage=arms, alpha=0.3, context_dimension=128)
        #policy = linucb.LinUCB(historystorage, modelstorage, actions, alpha=0.3, context_dimension=128)
    elif bandit == 'LinThompSamp':
        policy = linthompsamp.LinThompSamp(history_storage=historystorage, model_storage=modelstorage, action_storage=arms, delta=0.61, R=0.01, epsilon=0.71)
    elif bandit == 'Random':
        policy = 0
    elif bandit == 'Similar':
        policy = 0

    return policy    

