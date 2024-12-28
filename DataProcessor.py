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
