from sklearn.tree import DecisionTreeClassifier, _tree

# Fit the DecisionTreeClassifier
#mod = DecisionTreeClassifier().fit(X_train, array_indexes)

# Define function to find positive paths
def get_positive_paths(tree, feature_names, node=0, conditions=None):
    if conditions is None:
        conditions = {}
    
    left_child = tree.children_left[node]
    right_child = tree.children_right[node]
    threshold = tree.threshold[node]
    feature = tree.feature[node]
    
    # Check if it's a leaf node
    if left_child == _tree.TREE_LEAF and right_child == _tree.TREE_LEAF:
        # Check if it's a positive leaf node
        if tree.value[node][0, 1] > tree.value[node][0, 0]:  # More positive than negative
            print("Important Value Threshold Scenario:")
            # Print each feature's range in the conditions
            for feat, bounds in conditions.items():
                lower_bound = bounds.get('lower', '-∞')
                upper_bound = bounds.get('upper', '∞')
                print(f" - {feat}: {lower_bound} < {feat} <= {upper_bound}")
            print()
        return
    
    # Update bounds for the current feature in conditions
    feature_name = feature_names[feature]
    if left_child != _tree.TREE_LEAF:
        # Set an upper bound for the feature
        new_conditions = {k: v.copy() for k, v in conditions.items()}
        new_conditions.setdefault(feature_name, {}).update({'upper': threshold})
        get_positive_paths(tree, feature_names, left_child, new_conditions)
    
    if right_child != _tree.TREE_LEAF:
        # Set a lower bound for the feature
        new_conditions = {k: v.copy() for k, v in conditions.items()}
        new_conditions.setdefault(feature_name, {}).update({'lower': threshold})
        get_positive_paths(tree, feature_names, right_child, new_conditions)

# Define feature names list
feature_names = X_train.columns

# Extract and print paths for positive predictions
get_positive_paths(mod.tree_, feature_names)