import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import functools
import itertools
import sympy
from tqdm.notebook import trange,tqdm
import copy
import pickle
from IPython.display import display,clear_output
from random import choice
import os
from PIL import Image
import subprocess

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.decomposition import PCA, SparsePCA
from torch.autograd import grad
from scipy.optimize import minimize as scipy_min
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, Bounds, linprog
from sympy import lambdify
from sympy import Symbol as sb
from error_injection import MissingValueError, SamplingError, Injector
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score, auc, roc_curve, roc_auc_score, f1_score
np.random.seed(1)

class style():
    RED = '\033[31m'
    GREEN = '\033[32m'
    BLUE = '\033[34m'
    RESET = '\033[0m'
    
from ucimlrepo import fetch_ucirepo
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
    

def load_mpg():
    # fetch dataset
    auto_mpg = fetch_ucirepo(id=9)

    X = auto_mpg.data.features
    y = auto_mpg.data.targets
    
    # with this random seed, no null value is included in the test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train = copy.deepcopy(X_train).reset_index(drop=True)
    X_test = copy.deepcopy(X_test).reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test


# first impute the data and make it hypothetically clean
def load_mpg_cleaned():
    # fetch dataset
    auto_mpg = fetch_ucirepo(id=9)

    X = auto_mpg.data.features
    y = auto_mpg.data.targets
    
    # assumed gt imputation
    imputer = KNNImputer(n_neighbors=10)
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train = copy.deepcopy(X_train).reset_index(drop=True)
    X_test = copy.deepcopy(X_test).reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test

##################################################################################################
symbol_id = -1

def create_symbol(suffix=''):
    global symbol_id
    symbol_id += 1
    name = f'e{symbol_id}_{suffix}' if suffix else f'e{symbol_id}'
    return sympy.Symbol(name=name)
    
def impute_data(x):
    imputers = [KNNImputer(n_neighbors=5), KNNImputer(n_neighbors=10), KNNImputer(n_neighbors=15), IterativeImputer()]
    imputed_datasets = []
    for imp in imputers:
        imputed_datasets.append(imp.fit_transform(x))
    return imputed_datasets

def data_to_symbol(imputed_datasets):
    global symbol_id
    symbols_in_data = set()
    symbol_to_position = dict()
    symbolic_data = imputed_datasets[0].tolist()
    XS = imputed_datasets[0].tolist()
    XR = imputed_datasets[0].tolist()
    X_extended_max = np.array(imputed_datasets).max(axis=0)
    X_extended_min = np.array(imputed_datasets).min(axis=0)
    for row in range(len(symbolic_data)):
        for col in range(len(symbolic_data[0])):
            xmin = X_extended_min[row][col]
            xmax = X_extended_max[row][col]
            if xmin != xmax:
                xmean = (xmax + xmin) / 2
                xradius = (xmax - xmin) / 2
                new_symbol = create_symbol()
                symbolic_data[row][col] = xmean + xradius*new_symbol
                XS[row][col] = xradius*new_symbol
                XR[row][col] = xmean
                symbols_in_data.add(new_symbol)
                symbol_to_position[new_symbol] = (row, col)
            else:
                XS[row][col] = 0
    XS = sympy.Matrix(XS)
    XR = sympy.Matrix(XR)
    return symbolic_data, symbols_in_data, XS, XR
    
def fixed_point(XS, XR, y, symbolic_data):
    VT, sigma, V = np.linalg.svd(np.array((XR.T*XR).tolist()).astype(float))
    lamb = 0.1
    eigenvalues = 1 - sigma/len(symbolic_data)*2*lamb
    V_mat = sympy.Matrix(V)
    y_mat = sympy.Matrix(y)

    wR = (XR.T*XR).inv()*XR.T*y_mat
    wS_non_data = 0.0*V_mat.row(0).T
    for i in range(len(symbolic_data[0])):
        wS_non_data = wS_non_data + sb(f'k{i}')*sb(f'ep{i}')*V_mat.row(i).T
    A = V_mat.inv()*np.diag(eigenvalues)*V_mat
    wS_data = (np.identity(len(symbolic_data[0]))-A).inv()*((XS.T*XR + XR.T*XS)*wR - XS.T*y_mat)*lamb*2/len(symbolic_data)

    wS = wS_non_data + wS_data
    w = wS + wR
    w_prime = ((XS.T*XR + XR.T*XS + XS.T*XS)*wS + XS.T*XS*wR).expand()
    w_prime_projected = V_mat*w_prime
    
    eqs = []
    for d in range(len(symbolic_data[0])):
        eq1 = (1-eigenvalues[d])*sb(f'k{d}')
        eq2 = 0
        coef_dict = dict()
        coef_dict['const'] = 0
        for i in range(len(symbolic_data[0])):
            coef_dict[sb(f'k{i}')] = 0
        for arg in w_prime_projected[d].args:
            contain_k = False
            for i in range(len(symbolic_data[0])):
                symb_k = sb(f'k{i}')
                if symb_k in arg.free_symbols:
                    coef_dict[symb_k] = coef_dict[symb_k] + abs(arg.args[0])
                    contain_k = True
                    break
            if not(contain_k):
                coef_dict['const'] = coef_dict['const'] + abs(arg.args[0])
        eq2 = coef_dict['const']
        for i in range(len(symbolic_data[0])):
            eq2 = eq2 + sb(f'k{i}')*coef_dict[sb(f'k{i}')]
        eqs.append(sympy.Eq(eq1, eq2*lamb*2/len(symbolic_data)))
    
    result = sympy.solve(eqs, [sb(f'k{i}') for i in range(len(symbolic_data[0]))])
    param = wR+wS.subs(result)
    return param

# affine form to interval
def to_interval(expr):
    if not(isinstance(expr, sympy.Expr)):
        assert isinstance(expr, int) or isinstance(expr, float)
        return expr
        
    const_sum = 0
    coef_sum = 0
    
    if len(expr.args) == 0:
        return expr

    expr = expr.expand()
    for arg in expr.args:
        if arg.free_symbols:
            pair = arg.args
            # if pair is empty, then coef=1
            if pair:
                coef = pair[0]
            else:
                coef = 1
            coef_sum += abs(coef)
        else:
            const_sum += arg
        
    return [const_sum-coef_sum,const_sum+coef_sum]
    
def poly_to_interval(expr):
    max_expr, min_expr = 0, 0
    if not(expr.free_symbols):
        min_expr = max_expr = expr
        loss_quantiles.append([min_expr, max_expr])

    for arg in expr.args:
        if arg.free_symbols:
            p = arg.as_poly()
            monomial_exponents = p.monoms()[0]
            monomial = sympy.prod(x**k for x, k in zip(p.gens, monomial_exponents))
            coef = p.coeff_monomial(monomial)
            # overapproximation
            max_expr += abs(coef)
            min_expr += -abs(coef)
        else:
            max_expr += arg
            min_expr += arg
    return [min_expr,max_expr]


linearization_dict = dict()
reverse_linearization_dict = dict()
# merge high order errors and keep the constant and linear terms
def merge_high_order_components(expr_ls):
    processed_expr_ls = [0 for _ in range(len(expr_ls))]
    for expr_id, expr in enumerate(expr_ls):
        # Do not support monomial expr currently, e.g., expr = 1.5*e1.
        # At lease two monomials in expr, e.g., expr = 1.5*e1 + 2.
        if not(expr.free_symbols):
            processed_expr_ls[expr_id] += expr
            continue
        expr = expr.expand()
        for arg in expr.args:
            if not(arg.free_symbols):
                processed_expr_ls[expr_id] += arg
                continue
            p = arg.as_poly()
            monomial_exponents = p.monoms()[0]
            
            # only deal with non-linear monomials (order > 2)
            if sum(monomial_exponents) <= 1:
                processed_expr_ls[expr_id] += arg
                continue

            monomial = sympy.prod(x**k for x, k in zip(p.gens, monomial_exponents))
            # check global substitution dictionary
            if monomial in linearization_dict:
                processed_expr_ls[expr_id] += arg.coeff(monomial)*linearization_dict[monomial]
            else:
                found = False
                subs_monomial = create_symbol()
                for symb in monomial.free_symbols:
                    if symb in reverse_linearization_dict:
                        equivalent_monomial = monomial.subs(symb, reverse_linearization_dict[symb])
                        if equivalent_monomial in linearization_dict:
                            subs_monomial = linearization_dict[equivalent_monomial]
                            found = True
                            break
                linearization_dict[monomial] = subs_monomial
                if not(found):
                    reverse_linearization_dict[subs_monomial] = monomial
                processed_expr_ls[expr_id] += arg.coeff(monomial)*subs_monomial
                
    return processed_expr_ls


# take a list of expressions as input, output the list of monomials and generator vectors,
def get_generators(expr_ls):
    monomials = dict()
    for expr_id, expr in enumerate(expr_ls):
        if not(isinstance(expr, sympy.Expr)) or not(expr.free_symbols):
            continue
        expr = expr.expand()
        p = sympy.Poly(expr)
        monomials_in_expr = [sympy.prod(x**k for x, k in zip(p.gens, mon))
                             for mon in p.monoms() if sum(mon) >= 1]
        for monomial in monomials_in_expr:
            coef = float(p.coeff_monomial(monomial))
            if monomial in monomials:
                if len(monomials[monomial]) < expr_id:
                    monomials[monomial] = monomials[monomial] + [0 for _ in range(expr_id-len(monomials[monomial]))]
                monomials[monomial].append(coef)
            else:
                monomials[monomial] = [0 for _ in range(expr_id)] + [coef]

    for monomial in monomials:
        if len(monomials[monomial]) < len(expr_ls):
            monomials[monomial] = monomials[monomial] + [0 for _ in range(len(expr_ls)-len(monomials[monomial]))]
    
    return monomials


# take a list of expressions as input, output the ordered list of monomials,
# the first ones refer to ones with overall larger coefficients.
def heuristic_based_choice_merge_small(expr_ls):
    monomials = get_generators(expr_ls)
    monomial_heuristic_vals = dict()
    for monomial, monomial_coefs in monomials.items():
        monomial_norm = np.linalg.norm(monomial_coefs)
        monomial_heuristic_vals[monomial] = monomial_norm
    monomials_ordered = sorted(list(monomial_heuristic_vals.items()), reverse=True, key=lambda x: x[1])
    return monomials_ordered


# prune components with small coefs
# the components with coef less than pruning_ratio * maximum_coef will be merged together
# budget is the maximum number of terms that can be kept in each dimension
def merge_small_components(expr_ls, pruning_ratio=1e3, budget=10, strategy='budget'):
    symb_ordered = heuristic_based_choice_merge_small(expr_ls)
    if strategy != 'budget':
        raise NotImplementedError
    symb_keep = set([p[0] for p in symb_ordered[:budget]])
    symb_merge = set([p[0] for p in symb_ordered[budget:]])
    processed_expr_ls = []
    for expr_id, expr in enumerate(expr_ls):
        if not(isinstance(expr, sympy.Expr)) or not(expr.free_symbols):
            processed_expr_ls.append(expr)
            continue
        expr = expr.expand()
        processed_expr = 0
        merged_coef = 0
        for arg in expr.args:
            if arg.free_symbols:
                pair = arg.args
                # if pair is empty, then coef = 1
                if pair:
                    coef, err_symbol = pair[0], pair[1]
                else:
                    coef, err_symbol = 1, arg
                if err_symbol in symb_keep:
                    processed_expr += arg
                else:
                    merged_coef += abs(coef)
            else:
                processed_expr += arg
    
        processed_expr += merged_coef * create_symbol()
        processed_expr_ls.append(processed_expr)
        
    return processed_expr_ls


#def merge_small_components_pca(expr_ls, budget=10):
#    if not(isinstance(expr_ls, sympy.Expr)):
#        expr_ls = sympy.Matrix(expr_ls)
#    if expr_ls.free_symbols:
#        center = expr_ls.subs(dict([(symb, 0) for symb in expr_ls.free_symbols]))
#    else:
#        return expr_ls
#    monomials_dict = get_generators(expr_ls)
#    generators = np.array([monomials_dict[m] for m in monomials_dict])
#    if len(generators) <= budget:
#        return expr_ls
#    monomials = [m for m in monomials_dict]
#    pca = PCA(n_components=len(generators[0]))
#    pca.fit(np.concatenate([generators, -generators]))
#    transformed_generators = pca.transform(generators)
#    transformed_generator_norms = np.linalg.norm(transformed_generators, axis=1, ord=2)
#    # from largest to lowest norm
#    sorted_indices = transformed_generator_norms.argsort()[::-1].astype(int)
#    sorted_transformed_generators = transformed_generators[sorted_indices]
#    sorted_monomials = [monomials[idx] for idx in sorted_indices]
#    new_transformed_generators = np.concatenate([sorted_transformed_generators[:budget],
#                                                 np.diag(np.sum(np.abs(sorted_transformed_generators[budget:]),
#                                                                axis=0))])
#    new_generators = pca.inverse_transform(new_transformed_generators)
#    new_monomials = sorted_monomials[:budget] + [create_symbol() for _ in range(len(generators[0]))]
#    
#    processed_expr_ls = center
#    for monomial_id in range(len(new_monomials)):
#        processed_expr_ls += sympy.Matrix(new_generators[monomial_id])*new_monomials[monomial_id]
#    
#    return processed_expr_ls
    
def merge_small_components_pca(expr_ls, budget=10):
    if not(isinstance(expr_ls, sympy.Expr)):
        expr_ls = sympy.Matrix(expr_ls)
    if expr_ls.free_symbols:
        center = expr_ls.subs(dict([(symb, 0) for symb in expr_ls.free_symbols]))
    else:
        return expr_ls
    monomials_dict = get_generators(expr_ls)
    generators = np.array([monomials_dict[m] for m in monomials_dict])
    if len(generators) <= budget:
        return expr_ls
    monomials = [m for m in monomials_dict]
    
    VT, sigma, V = np.linalg.svd(np.matmul(np.concatenate([generators, -generators]).T,
                                           np.concatenate([generators, -generators])))
    transformed_generators = np.matmul(V, generators.T).T
    transformed_generator_norms = np.linalg.norm(transformed_generators, axis=1, ord=2)
    
    # from largest to lowest norm
    sorted_indices = transformed_generator_norms.argsort()[::-1].astype(int)
    sorted_transformed_generators = transformed_generators[sorted_indices]
    sorted_monomials = [monomials[idx] for idx in sorted_indices]
    new_transformed_generators = np.concatenate([sorted_transformed_generators[:budget],
                                                 np.diag(np.sum(np.abs(sorted_transformed_generators[budget:]), axis=0))])
    new_generators = np.matmul(VT, new_transformed_generators.T).T
    new_monomials = sorted_monomials[:budget] + [create_symbol() for _ in range(len(generators[0]))]
    
    processed_expr_ls = center
    for monomial_id in range(len(new_monomials)):
        processed_expr_ls += sympy.Matrix(new_generators[monomial_id])*new_monomials[monomial_id]
    
    return processed_expr_ls

def merge_small_components_v_selected(expr_ls, symbols_in_data, symbolic_in_orig_weight, V, VT):
    if not(isinstance(expr_ls, sympy.Expr)):
        expr_ls = sympy.Matrix(expr_ls)
    if expr_ls.free_symbols:
        center = expr_ls.subs(dict([(symb, 0) for symb in expr_ls.free_symbols]))
    else:
        return expr_ls
    
    monomials_dict = get_generators(expr_ls)
    generators = np.array([monomials_dict[m] for m in monomials_dict])

    monomials = [m for m in monomials_dict]

    transformed_generators = np.matmul(V, generators.T).T
    transformed_generator_norms = np.linalg.norm(transformed_generators, axis=1, ord=2)
    
    selected_indices = []
    not_selected_indices = []
    for idx, monomial in enumerate(monomials_dict.keys()):
        if (monomial in symbols_in_data) or (monomial in symbolic_in_orig_weight):
            selected_indices += [idx]
        else:
            not_selected_indices += [idx]
    selected_indices = np.array(selected_indices).astype(int)
    not_selected_indices = np.array(not_selected_indices).astype(int)

    selected_transformed_generators = transformed_generators[selected_indices]
    selected_monomials = [monomials[idx] for idx in selected_indices]
    
    not_selected_transformed_generators = transformed_generators[not_selected_indices]
    
    new_transformed_generators = np.concatenate([selected_transformed_generators,
                                                 np.diag(np.sum(np.abs(not_selected_transformed_generators),
                                                                axis=0))])

    new_generators = np.matmul(VT, new_transformed_generators.T).T
    new_monomials = selected_monomials + [create_symbol() for _ in range(len(generators[0]))]
    
    processed_expr_ls = center
    for monomial_id in range(len(new_monomials)):
        processed_expr_ls += sympy.Matrix(new_generators[monomial_id])*new_monomials[monomial_id]
    
    return processed_expr_ls, new_transformed_generators

def merge_independent_terms(expr_ls, initial_common_symbols=set()):
    # get a set of shared symbols
    common_symbols = copy.deepcopy(initial_common_symbols)
    common_symbols_cnt = dict()
    processed_expr_ls = [0 for _ in range(len(expr_ls))]
    for expr_id, expr in enumerate(expr_ls):
        if not(isinstance(expr, sympy.Expr)):
            assert isinstance(expr, int) or isinstance(expr, float)
        else:
            for symb in expr.free_symbols:
                if symb in common_symbols_cnt:
                    common_symbols_cnt[symb] += 1
                else:
                    common_symbols_cnt[symb] = 1
    for symb in common_symbols_cnt:
        if common_symbols_cnt[symb] > 1:
            common_symbols.add(symb)
    
    # prune each expression by merging independent terms (not shared with other expressions)
    # assume only linear and constant terms in the expression
    # replace each expression in the list with corresponding pruned version
    for expr_id, expr in enumerate(expr_ls):
        if len(expr.args) == 0:
            processed_expr_ls[expr_id] += expr
            continue
        expr = expr.expand()
        processed_expr = 0
        merged_coef = 0
        # go through each linear/constant term
        for arg in expr.args:
            if arg.free_symbols:
                pair = arg.args
                # if pair is empty, then coef=1
                if pair:
                    coef, err_symbol = pair[0], pair[1]
                else:
                    coef, err_symbol = 1, arg
                # keep if symbol is shared, otherwise merge
                if err_symbol in common_symbols:
                    processed_expr += arg
                else:
                    merged_coef += abs(coef)
            else:
                processed_expr += arg
        processed_expr += merged_coef * create_symbol()
        processed_expr_ls[expr_id] += processed_expr
    
    return processed_expr_ls


def get_vertices(affset):
    l = len(affset)
    distinct_symbols = set()
    for expr in affset:
        if not(isinstance(expr, sympy.Expr)):
            assert isinstance(expr, int) or isinstance(expr, float)
        else:
            if distinct_symbols:
                distinct_symbols = distinct_symbols.union(expr.free_symbols)
            else:
                distinct_symbols = expr.free_symbols
    distinct_symbols = list(distinct_symbols)
    # print(distinct_symbols)
    combs = [list(zip(distinct_symbols,list(l))) for l in list(itertools.product([-1, 1], repeat=len(distinct_symbols)))]
    res = set()
    for assignment in combs:
        res.add(tuple([expr.subs(assignment) for expr in affset]))
    return(res)


#def plot_conretiztion(affset, alpha = 0.5, color='red', budget=-1, line=False, style='-'):
#    if budget > -1:
#        affset = merge_small_components_pca(affset, budget=budget)
#    pts = np.array(list(map(list, get_vertices(affset))))
#    hull = ConvexHull(pts)
#    if line:
#        for simplex in hull.simplices:
#            #Draw a black line between each
#            plt.plot(pts[simplex, 0], pts[simplex, 1], linewidth=1, linestyle=style)
#    plt.fill(pts[hull.vertices,0], pts[hull.vertices,1],color,alpha=alpha)

def plot_conretiztion(affset, alpha = 0.5, color='red', budget=-1, line=False, style='-', lcolor='black'):
    if budget > -1:
        affset = merge_small_components_pca(affset, budget=budget)
    pts = np.array(list(map(list, get_vertices(affset))))
    hull = ConvexHull(pts)
    if line:
        for simplex in hull.simplices:
            #Draw a black line between each
            plt.plot(pts[simplex, 0], pts[simplex, 1], linewidth=1, color=lcolor, linestyle=style)
    plt.fill(pts[hull.vertices,0], pts[hull.vertices,1],color,alpha=alpha)


# Training functions and muti-process training for more than one instances of data.
def train_model(symbolic_data, symbols_in_data,  y, N = 75, lamb = 0.1, num_attrs = 6, merge_budget = 20):
#    save_path = os.getcwd()+'/figs'
#    save_path = save_path + f'/{len(symbolic_data)}_{num_attrs}_{lamb}_{merge_budget}'
#    os.mkdir(save_path)
#    current_path = os.getcwd()
    symbolic_data_mat = sympy.Matrix([t[:num_attrs] for t in symbolic_data])
    y_mat = sympy.Matrix(y)
    param = sympy.Matrix([0.0 for _ in range(num_attrs)])
    params = []
    
#    print(f'Trainning: \n data size = {len(symbolic_data)} \n N = {N} \n lamb = {lamb} \n nattrs = {num_attrs} \n merge_budget = {merge_budget}')
    
    t = tqdm(range(N), desc='Gradient Decent', leave=False)

    for iteration in t:
        preds_diff = symbolic_data_mat*param - y_mat
        grad = symbolic_data_mat.transpose()*preds_diff
        grad = merge_high_order_components(grad)
        param = [param[j] - grad[j]*lamb/len(symbolic_data) for j in range(len(param))]
        param = merge_independent_terms(merge_small_components_pca(param, budget=merge_budget),
                                        symbols_in_data)
        param = sympy.Matrix(param)
        params.append(param)
        
    return params

def train_model_visualize(symbolic_data, symbols_in_data, y, N = 75, val_epoch = 5, lamb = 0.1, num_attrs = 6, merge_budget = 20, para_fig_x = 3, para_fig_y = 7, param = [], param_samp = []):
    save_path = os.getcwd()+'/figs'
    save_path = save_path + f'/{len(symbolic_data)}_{num_attrs}_{lamb}_{merge_budget}'
    os.mkdir(save_path)
    current_path = os.getcwd()
    symbolic_data_mat = sympy.Matrix([t[:num_attrs] for t in symbolic_data])
    y_mat = sympy.Matrix(y)
    if len(param) <= 0:
        param = sympy.Matrix([0.0 for _ in range(num_attrs)])
    loss_quantiles = []
    params = []
    
    print(f'Trainning: \n data size = {len(symbolic_data)} \n N = {N} \n lamb = {lamb} \n nattrs = {num_attrs} \n merge_budget = {merge_budget}')
    
    
    t = tqdm(range(N), desc='Symbolic')
    
    fig = plt.figure(figsize=(10, 8))
    ax3 = fig.add_subplot(212)
    ax3.title.set_text('Sum of Coef')
    ax3.tick_params(axis='x', labelrotation=90)
    ax = fig.add_subplot(221)
    ax.title.set_text('MSE')
    ax2 = fig.add_subplot(222)
    ax2.title.set_text(f'Param {para_fig_x, para_fig_y}')
    dh = display(fig, display_id=True)

    for iteration in t:
        preds_diff = symbolic_data_mat*param - y_mat
        grad = symbolic_data_mat.transpose()*preds_diff
        grad = merge_high_order_components(grad)
        param = [param[j] - grad[j]*lamb/len(symbolic_data) for j in range(len(param))]
        param = merge_independent_terms(merge_small_components_pca(param, budget=merge_budget),
                                        symbols_in_data)
        param = sympy.Matrix(param)
        params.append(param)
        # record mse ranges
        if ((iteration + 1) % val_epoch == 0):
            preds_diff = symbolic_data_mat*param - y_mat
            mse = (preds_diff.transpose()*preds_diff/len(symbolic_data)).expand()[0]
            max_expr, min_expr = 0, 0
            if not(mse.free_symbols):
                min_expr = max_expr = mse
                loss_quantiles.append([min_expr, max_expr])
#                 print(mse.free_symbols)

            for arg in mse.args:
                if arg.free_symbols:
                    p = arg.as_poly()
                    monomial_exponents = p.monoms()[0]
                    monomial = sympy.prod(x**k for x, k in zip(p.gens, monomial_exponents))
                    coef = p.coeff_monomial(monomial)
                    # overapproximation
                    max_expr += abs(coef)
                    min_expr += -abs(coef)
                else:
                    max_expr += arg
                    min_expr += arg
            loss_quantiles.append([min_expr, max_expr])
           
            loss_bounds = np.array(loss_quantiles).astype(float)
            ax.clear()
            ax.plot([val_epoch*i for i in range(1, len(loss_quantiles)+1)], loss_bounds[:, 1], label='upper bound')
            ax.plot([val_epoch*i for i in range(1, len(loss_quantiles)+1)], loss_bounds[:, 0], label='lower bound')
            ax.fill_between([val_epoch*i for i in range(1, len(loss_quantiles)+1)], loss_bounds[:, 0], loss_bounds[:, 1], color='b', alpha=.1, label='range of loss')
            
            ax2.clear()
            affset = merge_small_components_pca([param[para_fig_x],param[para_fig_y]], budget=5)
            pts = np.array(list(map(list, get_vertices(affset))))
            hull = ConvexHull(pts)
            ax2.fill(pts[hull.vertices,0], pts[hull.vertices,1],'yellow',alpha=0.7)
            if len(param_samp) > 0:
                ax2.scatter([sp[para_fig_x] for sp in param_samp], [sp[para_fig_y] for sp in param_samp], s=0.75)
            
            ax3.clear()
            histo = dict()
            for p in param:
                for s in p.free_symbols:
                    histo[s] = 0
            for p in param:
                for s in histo:
                    histo[s] += abs(p.coeff(s))
            symbs = sorted(list(histo.keys()), key=lambda sb: int(''.join(c for c in sb.name if c.isdigit())))
            vals = [histo[s] for s in symbs]
            symbs = [s.name for s in symbs]
            ax3.bar(symbs, vals, color ='grey')
            if iteration+1 < N:
                dh.update(fig)
            else:
                pintv = [to_interval(p) for p in param]
                dh.update(pintv)
            fig.savefig(save_path+f'/{iteration}.png')
#    return params, loss_quantiles, sample_params, sample_mses
    return params, loss_quantiles



# Training functions and muti-process training for more than one instances of data.
def train_model_adaptive_lr(symbolic_data, y, N = 75, val_epoch = 5, lamb = 0.1, num_attrs = 6,
                            merge_budget = 20, plot_mse = False, imputed_datasets = [],
                            para_fig_x = 3, para_fig_y = 7):
    save_path = os.getcwd()+'/figs'
    save_path = save_path + f'/{len(symbolic_data)}_{num_attrs}_{lamb}_{merge_budget}'
    os.mkdir(save_path)
    current_path = os.getcwd()
    symbolic_data_mat = sympy.Matrix([t[:num_attrs] for t in symbolic_data])
    y_mat = sympy.Matrix(y)
    param = sympy.Matrix([0.0 for _ in range(num_attrs)])
    loss_quantiles = []
    params = []
    nsample = 1000
    sample_mses = []
    sample_params = []
    train_sample = len(imputed_datasets) > 0
    
    print(f'Trainning: \n data size = {len(symbolic_data)} \n N = {N} \n lamb = {lamb} \n nattrs = {num_attrs} \n merge_budget = {merge_budget}')
    
    if train_sample:
        for s in tqdm(range(nsample), desc='Samples'):
            samp = sample_data(imputed_datasets)
            mses = []
            params = []
            w = np.zeros(num_attrs).reshape(1, -1)
            for i in range(N):
            # still compute mse on original clean data instead of sampled dirty data
                mses.append(((np.matmul(samp, w.T)-y.reshape(-1, 1))**2).mean())
                grad = (samp * (np.matmul(samp, w.T)-y.reshape(-1, 1)) * lamb).mean(axis=0)
                w = w - grad
                params.append(w[0])
            sample_mses.append(mses)
            sample_params.append(params)
    
    sample_params_np = np.array(sample_params)
    
    t = tqdm(range(N), desc='Symbolic')
    
    fig = plt.figure(figsize=(10, 4))
    if plot_mse:
        ax = fig.add_subplot(121)
    if train_sample:
        ax2 = fig.add_subplot(122)
    dh = display(fig, display_id=True)

    for iteration in t:
        preds_diff = symbolic_data_mat*param - y_mat
        grad = symbolic_data_mat.transpose()*preds_diff/len(symbolic_data)
        grad = merge_high_order_components(grad)
        # only apply adpative lr after certain iters
        if iteration > starting_iter:
            lr = choose_learning_rate(param, grad)
            print(lr)
        else:
            lr = lamb
        param = [param[j] - grad[j]*lr for j in range(len(param))]
#         param = [param[j] - grad[j]*lamb/len(symbolic_data) for j in range(len(param))]
        param = merge_independent_terms(merge_small_components_pca(param, budget=merge_budget),
                                        symbols_in_data)
        param = sympy.Matrix(param)
        params.append(param)
        # record mse ranges
        if ((iteration + 1) % val_epoch == 0):
            preds_diff = symbolic_data_mat*param - y_mat
            mse = (preds_diff.transpose()*preds_diff/len(symbolic_data)).expand()[0]
            max_expr, min_expr = 0, 0
            if not(mse.free_symbols):
                min_expr = max_expr = mse
                loss_quantiles.append([min_expr, max_expr])

            for arg in mse.args:
                if arg.free_symbols:
                    p = arg.as_poly()
                    monomial_exponents = p.monoms()[0]
                    monomial = sympy.prod(x**k for x, k in zip(p.gens, monomial_exponents))
                    coef = p.coeff_monomial(monomial)
                    # overapproximation
                    max_expr += abs(coef)
                    min_expr += -abs(coef)
                else:
                    max_expr += arg
                    min_expr += arg
            loss_quantiles.append([min_expr, max_expr])
            if plot_mse:
                loss_bounds = np.array(loss_quantiles).astype(float)
                ax.clear()
                ax.plot([val_epoch*i for i in range(1, len(loss_quantiles)+1)], loss_bounds[:, 1], label='upper bound')
                ax.plot([val_epoch*i for i in range(1, len(loss_quantiles)+1)], loss_bounds[:, 0], label='lower bound')
                ax.fill_between([val_epoch*i for i in range(1, len(loss_quantiles)+1)],
                                loss_bounds[:, 0], loss_bounds[:, 1],
                                color='b', alpha=.1, label='range of loss')
            if train_sample:
                ax2.clear()
                affset = merge_small_components_pca([param[para_fig_x],param[para_fig_y]], budget=5)
                pts = np.array(list(map(list, get_vertices(affset))))
                hull = ConvexHull(pts)
                ax2.fill(pts[hull.vertices,0], pts[hull.vertices,1],'yellow',alpha=0.7)
                ax2.scatter(sample_params_np[:,iteration,para_fig_x], sample_params_np[:,iteration,para_fig_y], s=1)
            if iteration+1 < N:
                dh.update(fig)
            else:
                pintv = [to_interval(p) for p in param]
                dh.update(pintv)
            fig.savefig(save_path+f'/{iteration}.png')
    if train_sample:
        return params, loss_quantiles, sample_params, sample_mses
    return params, loss_quantiles


def plot_mse(loss_quantiles, val_epoch):
    loss_bounds = np.array(loss_quantiles).astype(float)
    fig, ax = plt.subplots()
    ax.plot([val_epoch*i for i in range(1, len(loss_quantiles)+1)], loss_bounds[:, 1], label='upper bound')
    ax.plot([val_epoch*i for i in range(1, len(loss_quantiles)+1)], loss_bounds[:, 0], label='lower bound')
    ax.fill_between([val_epoch*i for i in range(1, len(loss_quantiles)+1)],
                    loss_bounds[:, 0], loss_bounds[:, 1],
                    color='b', alpha=.1, label='range of loss')
    plt.legend()
    plt.xlabel('Epoch num')
    plt.ylabel('MSE loss')

    plt.show()

def plot_param_time_lapse(params, start, end, gap=1, x=0, y=1, budget=5):
    fig = plt.figure(4)
    para0 = params[-1]
    para0 = merge_small_components_pca(para0, budget=budget)
    plot_conretiztion([para0[x], para0[y]], 0.5)
    for i in range (start, end, gap):
        parai = params[i]
        plot_conretiztion([parai[x], parai[y]], 0.05, 'blue',budget=budget)
    plt.xlabel(f'$w_{x}$', fontsize=16)
    plt.ylabel(f'$w_{y}$', fontsize=16)
#     fig.suptitle('Pair-wise Concretization', fontsize=16)
    plt.show()
    
def plot_zono_and_sample(sample_params, sample_mses, x, y):
#     np.array(sample_params).transpose().shape
#     plt.subplot()
#     plot_conretiztion([new_params[-1][3], new_params[-1][7]], color='yellow',budget=5)
#     plt.scatter(all_param[:,3], all_param[:,7], s=1)
    return
    
def sample_data(imputed_datasets, uncert_inds=[], seed=42):
    imp_np = np.array(imputed_datasets)
    if len(uncert_inds) == 0:
        uncert_inds = list(itertools.product(range(imp_np.shape[1]),range(imp_np.shape[2])))
    np.random.seed(seed)
    choices = np.random.choice(np.arange(imp_np.shape[0]), len(uncert_inds), replace=True)
    sample_result = imputed_datasets[0].copy()
    for i, ind in enumerate(uncert_inds):
        sample_result[ind[0]][ind[1]] = imputed_datasets[choices[i]][ind[0]][ind[1]]
    return sample_result
    
class all_pw():
    def __init__(self, imputed, x_extended):
        self.imputed = imputed
        self.uncert_inds = np.argwhere(np.isnan(x_extended))
        self.numit = pow(len(self.imputed),len(self.uncert_inds))
        inds = list(itertools.repeat(list(range(len(self.imputed))), len(self.uncert_inds)))
        self.pw_iter = itertools.product(*inds)
        
    def next_pw(self):
        nxt = next(self.pw_iter, False)
        if nxt:
            ret = self.imputed[0].copy()
            for i in range(len(self.uncert_inds)):
                ret[self.uncert_inds[i][0]][self.uncert_inds[i][1]] = self.imputed[nxt[i]][self.uncert_inds[i][0]][self.uncert_inds[i][1]]
            return ret
        return False
        
def ground_truth_fixed_point(imputed_datasets, y, x_extended, mse = False):
    sample_params = []
    sample_mses = []
    pws = all_pw(imputed_datasets, x_extended)
    nsamp = pws.numit
    for s in tqdm(range(nsamp), desc='All PWs', leave=False):
        samp = pws.next_pw()
        assert(samp.any())
        sample_params.append(np.matmul(np.linalg.inv(np.matmul(samp.T, samp)), np.matmul(samp.T, y)))
        if mse:
            sample_mses.append(((np.matmul(samp, np.transpose(sample_params))-y.reshape(-1, 1))**2).mean())

    sample_params_np = np.array(sample_params)
    if mse:
        sample_mses_np = np.array(sample_mses)
        return sample_params_np, sample_mses_np
    return sample_params_np
            

def choose_learning_rate(param, grad, lr_lb=0.001, lr_ub=1):
    lr = sb('lr')
    param_monomials_dict = get_generators(param)
    grad_monomials_dict = get_generators(grad)
    sum_squared_norm = 0
    for monomial in set(param_monomials_dict.keys()).union(grad_monomials_dict.keys()):
        if monomial in param_monomials_dict:
            if monomial in grad_monomials_dict:
                new_generator_part1 = np.array(param_monomials_dict[monomial])
                new_generator_part2 = np.array(grad_monomials_dict[monomial])
                sum_squared_norm += (lr**2)*np.matmul(new_generator_part2.T, new_generator_part2)
                sum_squared_norm += (-2*lr)*np.matmul(new_generator_part1.T, new_generator_part2)
        else:
            new_generator = np.array(grad_monomials_dict[monomial])
            sum_squared_norm += (lr**2)*np.matmul(new_generator.T, new_generator)
        
    obj_func = lambdify([lr], sum_squared_norm)
    res = scipy_min(obj_func, lr_lb, bounds=Bounds(lr_lb, lr_ub))
    return res.x[0]
    
def sample_fixed_point(imputed_datasets, y, samp=10000, mse = False):
    sample_params = []
    sample_mses = []
    for s in tqdm(range(samp), desc='Samples', leave=False):
        samp = sample_data(imputed_datasets, seed=s)
        sample_params.append(np.matmul(np.linalg.inv(np.matmul(samp.T, samp)), np.matmul(samp.T, y)))
        if mse:
            sample_mses.append(((np.matmul(samp, np.transpose(sample_params))-y.reshape(-1, 1))**2).mean())

    sample_params_np = np.array(sample_params)
    if mse:
        sample_mses_np = np.array(sample_mses)
        return sample_params_np, sample_mses_np
    return sample_params_np
 
# type: s, fp, gd
# Metrics: range, mse
def get_metric(type, data, metric="range", symbolic_data = [], y = []):
    if metric == "range":
        if type == "s":
            max = np.array(data).max(axis=0)
            min = np.array(data).min(axis=0)
            return np.average(max-min)
        elif type == "fp":
            return np.average([(to_interval(p)[1]-to_interval(p)[0]) for p in data])
        elif type == "gd":
            return np.average([(to_interval(p)[1]-to_interval(p)[0]) for p in data[-1]])
    elif metric == "mse":
        if type == "s":
            max = np.array(data).max(axis=0)
            min = np.array(data).min(axis=0)
            return [(min+max)/2, min, max]
        elif type == "fp":
            symbolic_data_mat = sympy.Matrix([t[:len(data)] for t in symbolic_data])
            y_mat = sympy.Matrix(y)
            preds_diff = symbolic_data_mat*data - y_mat
            mse = (preds_diff.transpose()*preds_diff/len(symbolic_data)).expand()[0]
            int_mse = poly_to_interval(mse)
#            return abs(int_mse[1]-int_mse[0])
            return [(abs(int_mse[1])+abs(int_mse[0]))/2, abs(int_mse[0]), abs(int_mse[1])]
        elif type == "gd":
            symbolic_data_mat = sympy.Matrix([t[:num_attrs] for t in symbolic_data])
            y_mat = sympy.Matrix(y)
            preds_diff = symbolic_data_mat*data[-1] - y_mat
            mse = (preds_diff.transpose()*preds_diff/len(symbolic_data)).expand()[0]
            int_mse = poly_to_interval(mse)
            return [(int_mse[1]+int_mse[0])/2, int_mse[0], int_mse[1]]

def writetofile(fn, content):
    with open(fn,"w+") as f:
        f.write(content)
        f.close()

def plotBars(data, keys=[], xtics=[], dname = "", xl = "x", yl = "y"):
    save_path = os.getcwd()
    if dname != "":
        save_path = save_path + f'/{dname}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    fn = save_path + "/data"
    data = np.transpose(np.array(data))
    
    assert(len(data.shape)==2)
    assert(data.shape[1] == len(keys))
    assert(data.shape[0]== len(xtics))
    
    xrange = len(xtics)
    yrange = np.max(data)
    
    csv_str = "test\t"
    csv_str = csv_str + "\t".join(keys)
    
    for i in range(len(xtics)):
        line_str = str(xtics[i])
        for d in data[i]:
            line_str = line_str + "\t" + str(d)
        csv_str = csv_str + "\n" + line_str
    
    writetofile(fn+".csv", csv_str)
        
    with open("%s.gp"%fn, "w+") as file:
        file.write("\n".join([
            "set size ratio 0.4",
            "set terminal postscript color enhanced",
            "set output '%s.ps'"%(fn),
            "unset title",
            "set tmargin -3",
            "set bmargin -2",
            "set rmargin 0",
            "set lmargin 12",
            "set border 3 front linetype -1 linewidth 1.000",
            "set boxwidth 0.95 absolute",
            "set style fill solid 1.00 noborder",
            'set linetype 1 lw 1 lc rgb "#222222"',
            'set linetype 2 lw 1 lc rgb "#FF0000"',
            'set linetype 3 lw 1 lc rgb "#FFDD11"',
            'set linetype 4 lw 1 lc rgb "#0000FF"',
            'set linetype 5 lw 1 lc rgb "#55FF95"',
            'set linetype 6 lw 1 lc rgb "#55AAAA"',
            "set linetype cycle 4",
            
            "set grid nopolar",
            "set grid noxtics nomxtics ytics nomytics noztics nomztics nox2tics nomx2tics noy2tics nomy2tics nocbtics nomcbtics",
            "set grid layerdefault linetype 0 linewidth 3.000,  linetype 0 linewidth 1.000",
            
            "set key nobox autotitles columnhead Left reverse left",
            'set key font "Arial,26"',
            "set key width 1",
            "set key samplen 1",
            "set key spacing 1",
            "set key maxrows 2",
            "set key at 0, %s"%str(yrange),
            "set style histogram clustered gap 1 title  offset character 2, -0.25, 1",
            "set datafile missing '-'",
            "set style data histograms",
            
            'set xlabel font "Arial,34"',
            'set xlabel "%s"'%(xl),
            "set xlabel  offset character 0, -1, 0  norotate",

            "set xtics border in scale 0,0 nomirror   offset character 0.5, -0.5, 2 autojustify",
            'set xtics norangelimit font ",24"',
            "set xtics   ()",
            
            "set xrange [ -1 : %s]"%str(xrange),
                
            'set ylabel "%s"'%(yl),
            'set ylabel font "Arial,30"',
            "set ylabel offset character -3, 0, 0",
            
            "set ytics border in scale 0,0 mirror norotate  offset character 0, 0, 0 autojustify",
            'set ytics font ",34"',

            "set yrange [ 0 : %s ]"%str(yrange),
                
            "plot '%s.csv' using 2 t col, '' using 3:xtic(1) t col, '' using 4 t col"%fn
        ]))
        file.close()
        subprocess.call(["gnuplot", "%s.gp"%fn])
        subprocess.call(["ps2pdf", "%s.ps"%fn, "%s.pdf"%fn])
#        subprocess.call(["rm", "%s.gp"%fn])
        display(Image.open("%s.ps"%fn).rotate(-90,expand=True))
        subprocess.call(["rm", "%s.ps"%fn])
        return "%s.pdf"%fn

#def plotError(data, keys=[], xtics=[], dname = "", xl = "x", yl = "y"):
#    save_path = os.getcwd()
#    if dname != "":
#        save_path = save_path + f'/{dname}'
#    if not os.path.exists(save_path):
#        os.mkdir(save_path)
#    fn = save_path + "/data"
#    data = np.array(data)
#    
#    print(data.shape)
#    
#    assert(len(data.shape)==3)
#    assert(data.shape[0] == len(keys))
#    assert(data.shape[1]== len(xtics))
#    assert(data.shape[2]== 3)
#    
#    xlb = np.min(xtics)
#    xub = np.max(xtics)*1.1
#    yrange = np.max(data)
#    
##     csv_str = "test\t"
##     csv_str = csv_str + "\t".join(keys)
#    
#    for i in range(len(keys)):
#        f_str = "test\tval\tmin\tmax"
#        for d in range(len(xtics)):
#            line_str = str(xtics[d])
#            for m in data[i][d]:
#                line_str = line_str + "\t" + str(m)
#            f_str = f_str + "\n" + line_str
#    
#        writetofile(fn+ "_%s.csv"%keys[i], f_str)
#        
#    plot_cmd = ["'%s_%s.csv' using 2:3:4:xtic(1) title '%s'"%(fn,kn,kn) for kn in keys]
#    plot_cmd = ", ".join(plot_cmd)
#        
#    with open("%s.gp"%fn, "w+") as file:
#        file.write("\n".join([
#            "set size ratio 0.5",
#            "set terminal postscript color enhanced",
#            "set output '%s.ps'"%(fn),
#            "unset title",
#            "set tmargin -3",
#            "set bmargin -2",
#            "set rmargin 0",
#            "set lmargin 12",
#            "set border 3 front linetype -1 linewidth 1.000",
## #             "set boxwidth 0.95 absolute",
##             "set style fill solid 1.00 noborder",
##             'set linetype 1 lw 1 lc rgb "#222222"',
##             'set linetype 2 lw 1 lc rgb "#FF0000"',
##             'set linetype 3 lw 1 lc rgb "#FFDD11"',
##             'set linetype 4 lw 1 lc rgb "#0000FF"',
##             'set linetype 5 lw 1 lc rgb "#55FF95"',
##             'set linetype 6 lw 1 lc rgb "#55AAAA"',
##             "set linetype cycle 4",
#            
#            "set grid nopolar",
#            "set grid noxtics nomxtics ytics nomytics noztics nomztics nox2tics nomx2tics noy2tics nomy2tics nocbtics nomcbtics",
#            "set grid layerdefault linetype 0 linewidth 3.000,  linetype 0 linewidth 1.000",
#            
#            "set key nobox autotitles columnhead Left reverse left",
#            'set key font "Arial,26"',
#            "set key width 1",
#            "set key samplen 1",
#            "set key spacing 1",
#            "set key maxrows 2",
##             "set key at 0, %s"%str(yrange),
#            "set style histogram clustered gap 1 errorbars",
#            "set datafile missing '-'",
#            "set style data histograms",
#            
#            'set xlabel font "Arial,34"',
#            'set xlabel "%s"'%(xl),
#            "set xlabel  offset character 0, -1, 0  norotate",
#
#            "set xtics border in scale 0,0 nomirror offset character 0.5, -0.5, 2 autojustify",
#            'set xtics norangelimit font ",24"',
#            "set xtics   ()",
#            
## #             "set auto x",
##             "set xrange [ %s : %s]"%(str(xlb),str(xub)),
#                
#            'set ylabel "%s"'%(yl),
#            'set ylabel font "Arial,30"',
#            "set ylabel offset character -3, 0, 0",
#            
#            "set ytics border in scale 0,0 mirror norotate  offset character 0, 0, 0 autojustify",
#            'set ytics font ",34"',
#
##             "set yrange [ 0 : %s ]"%str(yrange),
#                
##             "plot '%s_1k.csv' using 2:3:4:xtic(1) title '1k'"%(fn)
#
#            'set style fill solid border rgb "black"',
#            "set yrange [0:*]",
#
#            "plot %s"%plot_cmd
#        ]))
#        file.close()
#        subprocess.call(["gnuplot", "%s.gp"%fn])
#        subprocess.call(["ps2pdf", "%s.ps"%fn, "%s.pdf"%fn])
##        subprocess.call(["rm", "%s.gp"%fn])
#        display(Image.open("%s.ps"%fn).rotate(-90,expand=True))
#        subprocess.call(["rm", "%s.ps"%fn])
#        return "%s.pdf"%fn

def plotError(data, keys=[], xtics=[], dname = "", xl = "x", yl = "y"):
    save_path = os.getcwd()
    if dname != "":
        save_path = save_path + f'/{dname}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    fn = save_path + "/data"
    data = np.array(data)
    
    print(data.shape)
    
    assert(len(data.shape)==3)
    assert(data.shape[0] == len(keys))
    assert(data.shape[1]== len(xtics))
    assert(data.shape[2]== 3)
    
    xlb = np.min(xtics)
    xub = np.max(xtics)*1.1
    yrange = np.max(data)
    
#     csv_str = "test\t"
#     csv_str = csv_str + "\t".join(keys)

    xtic_str = []
    
    for i in range(len(keys)):
        f_str = "test\tval\tmin\tmax"
        for d in range(len(xtics)):
            line_str = str(d)
            xtic_str.append('"%s" %s'%(str(xtics[d]),str(d)))
            for m in data[i][d]:
                line_str = line_str + "\t" + str(m)
            f_str = f_str + "\n" + line_str
    
        writetofile(fn+ "_%s.csv"%keys[i], f_str)
        
    xtic_str = ", ".join(xtic_str)
                            
    plot_cmd = ["'%s_%s.csv' using 2:3:4:xtic(1) title '%s'"%(fn,kn,kn) for kn in keys]
    plot_cmd = ", ".join(plot_cmd)
        
    with open("%s.gp"%fn, "w+") as file:
        file.write("\n".join([
            "set size ratio 0.4",
            "set terminal postscript color enhanced",
            "set output '%s.ps'"%(fn),
            "unset title",
            "set tmargin -3",
            "set bmargin -2",
            "set rmargin 0",
            "set lmargin 12",
            "set border 3 front linetype -1 linewidth 1.000",
            "set boxwidth 0.14",
            "set style fill solid",
            'set style line 1 lt 1 lc rgb "#85bcff" lw 5',
            'set style line 2 lt 2 lc rgb "#b5001a" lw 5',
            'set style line 3 lt 3 lc rgb "#00b200" lw 5',
            'set style line 4 lt 3 lc rgb "magenta" lw 5',
            'set style line 5 lt 3 lc rgb "#bd4c00" lw 5',
            'set style line 6 lt 3 lc rgb "#00b200" lw 5',
            'set style line 7 lt 3 lc rgb "#856ab0" lw 5',
            'set style line 8 lt 3 lc rgb "#85bcff" lw 5',
            
            "set grid nopolar",
            "set grid noxtics nomxtics ytics nomytics noztics nomztics nox2tics nomx2tics noy2tics nomy2tics nocbtics nomcbtics",
            "set grid layerdefault linetype 0 linewidth 3.000,  linetype 0 linewidth 1.000",
            
            "set key nobox autotitles columnhead Left reverse left",
            'set key font "Arial,26"',
            "set key width 1",
            "set key samplen 1",
            "set key spacing 1",
            "set key maxrows 2",
#             "set key at 0, %s"%str(yrange),
            "set style histogram clustered gap 1 errorbars",
            "set datafile missing '-'",
            "set style data histograms",
            
            'set xlabel font "Arial,34"',
            'set xlabel "%s"'%(xl),
            "set xlabel  offset character 0, -1, 0  norotate",

            "set xtics border in scale 0,0 nomirror offset character 0.5, -0.5, 2",
            'set xtics norangelimit font ",26"',
            "set xtics   ()",
            
# #             "set auto x",
            "set xrange [ -0.7 : %s]"%(str(len(xtics)-0.3)),
            'set xtics scale 0 (%s)'%(xtic_str),
                
            'set ylabel "%s"'%(yl),
            'set ylabel font "Arial,30"',
            "set ylabel offset character -3, 0, 0",
            
            "set ytics border in scale 0,0 mirror norotate  offset character 0, 0, 0 autojustify",
            'set ytics font ", 28" format "%.2f"',

            "set yrange [*:*]",
                
#             "plot '%s_1k.csv' using 2:3:4:xtic(1) title '1k'"%(fn)

#             'set style fill solid border rgb "black"',

#             "plot %s"%plot_cmd
            "plot '%s_%s.csv' using ($1-0.2):2:4:3:2 with candlesticks lt 1 lw 7 title '%s' whiskerbars, \
                '%s_%s.csv' using 1:2:4:3:2 with candlesticks lt 2 lw 7 title '%s' whiskerbars, \
                '%s_%s.csv' using ($1+0.2):2:4:3:2 with candlesticks lt 3 lw 7 title '%s' whiskerbars,"%(fn,keys[0],keys[0],fn,keys[1],keys[1],fn,keys[2],keys[2])
        ]))
        file.close()
        subprocess.call(["gnuplot", "%s.gp"%fn])
        subprocess.call(["ps2pdf", "%s.ps"%fn, "%s.pdf"%fn])
#        subprocess.call(["rm", "%s.gp"%fn])
        display(Image.open("%s.ps"%fn).rotate(-90,expand=True))
        subprocess.call(["rm", "%s.ps"%fn])
        return "%s.pdf"%fn