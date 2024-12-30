from scipy.sparse import csr_matrix
import numpy as np

class DataProcessor(object):

    @staticmethod
    def n_users(ratings):
        return len(ratings['user'].unique())

    @staticmethod
    def n_items(ratings):
        return len(ratings['item'].unique())

    @staticmethod
    def construct_real_matrix(ratings, processed, item_based=False):
        n_users = DataProcessor.n_users(ratings)
        n_items = DataProcessor.n_items(ratings)

        if item_based:

            return csr_matrix((processed.rating,
                           (processed.item, processed.user)),
                          shape=(n_items, n_users),
                          dtype='float32')

        return csr_matrix((processed.rating,
                           (processed.user, processed.item)),
                          shape=(n_users, n_items),
                          dtype='float32')

    @staticmethod
    def construct_one_valued_matrix(ratings, processed, item_based=False):
        n_users = DataProcessor.n_users(ratings)
        n_items = DataProcessor.n_items(ratings)

        if item_based:

            return csr_matrix((np.ones_like(processed.item.values),
                           (processed.item, processed.user)),
                          shape=(n_items, n_users),
                          dtype='float32')

        return csr_matrix((np.ones_like(processed.user.values),
                           (processed.user.values, processed.item.values)),
                          shape=(n_users, n_items),
                          dtype='float32')

    @staticmethod
    def construct_ratio_valued_matrix(ratings, processed, item_based=False):
        n_users = DataProcessor.n_users(ratings)
        n_items = DataProcessor.n_items(ratings)
        return csr_matrix((processed.rating / ratings.rating.max(),
                           (processed.user, processed.item)),
                          shape=(n_users, n_items),
                          dtype='float32')


class PreprocessDataset:
    @staticmethod
    def generate_internal_ids(all_set, with_dict=False):
        """
        Map new internal IDs for all users and items.

        Parameters:
        - all_set (DataFrame): The dataset containing 'user_id' and 'item_id'.
        - with_dict (bool): Whether to return the mapping dictionaries.

        Returns:
        - DataFrame: The dataset with updated 'user_id' and 'item_id'.
        - (Optional) dict: Mapping dictionary for user IDs.
        - (Optional) dict: Mapping dictionary for item IDs.
        """
        # Extract unique user and item IDs
        u_ids = all_set['user_id'].unique().tolist()
        i_ids = all_set['item_id'].unique().tolist()

        # Create mapping dictionaries
        user_dict = dict(zip(u_ids, range(len(u_ids))))
        item_dict = dict(zip(i_ids, range(len(i_ids))))

        # Map new IDs to the dataset
        all_set['user_id'] = all_set['user_id'].map(user_dict)
        all_set['item_id'] = all_set['item_id'].map(item_dict)

        # Return dataset and optionally the dictionaries
        if with_dict:
            return all_set, user_dict, item_dict
        return all_set
