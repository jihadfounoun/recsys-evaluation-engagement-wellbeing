import numpy as np
import pandas as pd


###############################################
# ITEM POPULARITY (calcolata su URM_train)
###############################################

def compute_item_popularity(URM_train):
    """
    Returns an array of item popularity counts based on URM_train.
    """
    URM_csc = URM_train.tocsc()
    item_popularity = np.ediff1d(URM_csc.indptr).astype(np.int32)
    return item_popularity


###############################################
# SERENDIPITY
###############################################



def compute_serendipity(recommendations_dict, item_popularity):
    """
    Calcola la serendipity media normalizzata tra 0 e 1.
    """
    n_users = len(recommendations_dict)
    total_serendipity = 0.0
    eps = 1e-10  # evita log(0)
    total_interactions = item_popularity.sum()

    # massimo possibile valore di -log2(p_i)
    max_ser_value = -np.log2(1 / total_interactions + eps)

    for user_id, rec_list in recommendations_dict.items():
        if len(rec_list) == 0:
            continue
        ser_u = 0.0
        for item_id in rec_list:
            p_i = item_popularity[item_id] / total_interactions
            ser_u += -np.log2(p_i + eps)
        total_serendipity += ser_u / len(rec_list)

    # normalizzazione tra 0 e 1
    return (total_serendipity / n_users) / max_ser_value


###############################################
# FAIRNESS — Group Exposure Fairness (4 groups)
###############################################

def compute_fairness(recommendations_dict, item_popularity, G=4):
    """
    Computes exposure fairness across G popularity groups.
    Based on Jain’s fairness index.

    recommendations_dict is: {user_id: [recommended_item_ids]}
    """
    # Group items into popularity bins
    popularity_bins = pd.qcut(item_popularity, q=G, labels=False, duplicates="drop")

    exposures = np.zeros(G)

    # Count exposures
    for user, rec_list in recommendations_dict.items():
        for item in rec_list:
            g = popularity_bins[item]
            exposures[g] += 1

    if exposures.sum() == 0:
        return 0.0

    exposures = exposures / exposures.sum()  # normalize

    # Jain fairness index
    fairness = (exposures.sum() ** 2) / (G * np.sum(exposures ** 2))
    return float(fairness)


def compute_recommendations_dict(recommender, urm_validation, cutoff=10):
    """
    Costruisce un dizionario user → recommended_items
    usando il metodo .recommend() del BaseRecommender.

    Parameters:
        recommender: modello già addestrato su URM_train
        URM_val: matrice di validazione (CSR)
        cutoff: lunghezza della lista di raccomandazioni

    Returns:
        recommendations_dict: {user_id: [item1, item2, ...]}
    """

    recommendations_dict = {}
    n_users = urm_validation.shape[0]

    for user_id in range(n_users):
        # Richiede una singola lista di item
        recs = recommender.recommend(
            user_id,
            cutoff=cutoff,
            remove_seen_flag=True,  # evita item del train
            remove_top_pop_flag=False,
            remove_custom_items_flag=False
        )
        recommendations_dict[user_id] = recs

    return recommendations_dict