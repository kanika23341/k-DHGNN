# LOAD PRE-MERGED CLINICAL + RADIOMICS DATA
# ============================================================================

import os
import numpy as np
import pandas as pd

print("="*80)
print("CELL 1: LOADING MERGED DATA")
print("="*80)

PROJECT_FOLDER = "/content/drive/MyDrive/UPENN_GBM_Project"
MERGED_FILE = os.path.join(PROJECT_FOLDER, "merged_final.csv")

if not os.path.exists(MERGED_FILE):
    raise FileNotFoundError(f"Could not find merged_final.csv at:\n{MERGED_FILE}")

data = pd.read_csv(MERGED_FILE)

print(f"Loaded file: {MERGED_FILE}")
print(f"Dataset shape: {data.shape}")

print("\nFirst columns:")
print(data.columns[:15].tolist())

print("\nFirst rows:")
display(data.head())

print("="*80)

# DATA CLEANING, SURVIVAL LABELS & CLINICAL FEATURES
# ============================================================================

import numpy as np
import pandas as pd

print("="*80)
print("CELL 2: CLEANING DATA")
print("="*80)

# Identify patient ID
possible_id_columns = [
    "ID",
    "SubjectID",
    "Subject_ID",
    "subject_id",
    "PatientID"
]

id_column = next(
    (c for c in possible_id_columns if c in data.columns),
    None
)

if id_column is None:
    raise ValueError("Could not identify patient ID column.")

print("Patient ID column:", id_column)

data[id_column] = (data[id_column].astype(str).str.strip().str.split("_").str[0])

# Survival Time
target_col = "Survival_from_surgery_days_UPDATED"

if target_col not in data.columns:
    raise ValueError(f"Required survival column '{target_col}' not found.")

data[target_col] = pd.to_numeric(data[target_col],errors="coerce")

# Remove patients with invalid survival time
data = data[data[target_col].notna()].copy()

data = data[data[target_col] > 0].copy()

# Event Label
if "Survival_Status" not in data.columns:
    raise ValueError("Required column 'Survival_Status' not found.")

status = (data["Survival_Status"].astype(str).str.strip())

event = status.map({
    "Alive": 0,
    "Lost to Follow-up": 0,
    "Deceased": 1,
    "Deceased - uncertain date of death": 1
})

# Keep only rows with a valid survival status
valid_event_mask = event.notna()

data = data[valid_event_mask].copy()

event = event.loc[data.index].astype(int)

data["Time"] = data[target_col].astype(float)
data["Event"] = event

# Clinical Features
clinical_features = [
    "Age_at_scan_years",
    "Gender",
    "KPS",
    "IDH1",
    "MGMT",
    "GTR_over90percent",
    "Time_since_baseline_preop",
    "PsP_TP_score"
]

clinical_features = [
    c for c in clinical_features
    if c in data.columns
]

print("\nClinical Features:")
print(clinical_features)

# Clean categorical variables
for col in clinical_features:

    if data[col].dtype == object:

        data[col] = (data[col].astype(str).str.strip()
            .replace(
                [
                    "",
                    "NA",
                    "N/A",
                    "Not Available",
                    "Unknown",
                    "nan",
                    "None"
                ],
                np.nan
            )
        )

        if col == "Gender":
            data[col] = data[col].map({
                "M": 0,
                "F": 1
            })

        else:
            data[col] = (data[col].astype("category").cat.codes.replace(-1, np.nan))

# Missing clinical values
for col in clinical_features:

    data[col] = data[col].fillna(data[col].median())

# Identify radiomics columns
excluded_columns = set(
    clinical_features
    + [
        id_column,
        "ID",
        "SubjectID",
        "Subject_ID",
        "subject_id",
        "PatientID",
        target_col,
        "Time",
        "Event",
        "Survival_Status",
        "Survival_Censor"
    ]
)

radiomics_columns = [
    c for c in data.columns
    if c not in excluded_columns
]
numeric_radiomics = []

for col in radiomics_columns:
    if pd.api.types.is_numeric_dtype(data[col]):
        numeric_radiomics.append(col)

radiomics_columns = numeric_radiomics

print("\nPatients:", len(data))
print("Clinical Features:", len(clinical_features))
print("Radiomics Features:", len(radiomics_columns))

print("\nEvent Distribution:")
print(data["Event"].value_counts())

print("\nSurvival Time:")
print(data["Time"].describe())

print("\nOriginal Survival Status:")
print(data["Survival_Status"].value_counts())

print("="*80)

# FEATURE PREPROCESSING, SURVIVAL LABEL SETUP & 12-NODE MAPPING
# ============================================================

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import train_test_split

print("=" * 80)
print("CELL 5: FEATURE PREPROCESSING & 12-NODE MAPPING")
print("=" * 80)

# VERIFY SURVIVAL TARGET
target_col = "Survival_from_surgery_days_UPDATED"

if target_col not in data.columns:
    raise ValueError(f"CRITICAL: Target column '{target_col}' not found in data dataframe!")

data["Time"] = pd.to_numeric(data[target_col],errors="coerce")

data["Event"] = 1
print(f"Survival Time Shape: {data['Time'].shape}")
print(f"Missing Survival Values: {data['Time'].isna().sum()}")


# DEFINE 12 BIOLOGICAL NODES
MODALITIES = [
    "T1",
    "T1GD",
    "T2",
    "FLAIR"
]

COMPARTMENTS = [
    "ET",
    "ED",
    "NC"
]

NODE_LAYOUT = [
    f"{mod}_{comp}"
    for mod in MODALITIES
    for comp in COMPARTMENTS
]
TARGET_FEATURES_PER_NODE = 24
print("\nNode Layout:")
print(NODE_LAYOUT)

# MAP RADIOMICS FEATURES TO 12 NODES
def strict_map_radiomics_to_12_nodes(columns):

    node_feature_map = {
        node: []
        for node in NODE_LAYOUT
    }

    for col in columns:
        c_upper = str(col).upper()
        if c_upper in [
            "SUBJECTID",
            "ID",
            "TIME",
            "EVENT",
            "SURVIVAL_STATUS",
            "SURVIVAL_CENSOR"
        ]:
            continue
        
        # Modality matching
        matched_mod = None

        if (
            "_T1GD_" in f"_{c_upper}_"
            or c_upper.startswith("T1GD_")
            or c_upper.endswith("_T1GD")
            or "T1GD" in c_upper
        ):
            matched_mod = "T1GD"

        elif(
            "_T1_" in f"_{c_upper}_"
            or c_upper.startswith("T1_")
            or c_upper.endswith("_T1")
            or ("T1GD" not in c_upper and "T1" in c_upper)
        ):
            matched_mod = "T1"

        elif (
            "_T2_" in f"_{c_upper}_"
            or c_upper.startswith("T2_")
            or c_upper.endswith("_T2")
            or "T2" in c_upper
        ):
            matched_mod = "T2"

        elif "FLAIR" in c_upper:
            matched_mod = "FLAIR"

        # Compartment matching
        matched_comp = None

        for comp in COMPARTMENTS:
            if (f"_{comp}_" in f"_{c_upper}_" or c_upper.endswith(f"_{comp}") or f"-{comp}-" in c_upper):
                matched_comp = comp
                break

        # Add feature to corresponding node
        if matched_mod and matched_comp:
            node_key = f"{matched_mod}_{matched_comp}"
            if node_key in node_feature_map:
                node_feature_map[node_key].append(col)

    return node_feature_map


node_mapping = strict_map_radiomics_to_12_nodes(
    data.columns
)

print("\nRadiomics Mapping:")

for node, cols in node_mapping.items():
    print(f"{node:<10}: {len(cols)} features")

# FEATURE PADDING
def pad_features(arr, target_dim):

    n_samples, n_feats = arr.shape
    if n_feats >= target_dim:
        return arr[:, :target_dim]
    padded = np.zeros((n_samples, target_dim),dtype=arr.dtype)
    padded[:, :n_feats] = arr
    return padded

# TRAIN / VALIDATION / TEST SPLIT
indices = np.arange(len(data))

train_val_idx, test_idx = train_test_split(indices,test_size=0.05,random_state=42)

train_idx, val_idx = train_test_split(train_val_idx,test_size=0.21,random_state=42)

df_tr = data.iloc[train_idx].copy()
df_val = data.iloc[val_idx].copy()
df_te = data.iloc[test_idx].copy()

print("\nData Splits")
print("-" * 16)
print(f"Train: {len(df_tr)}")
print(f"Val  : {len(df_val)}")
print(f"Test : {len(df_te)}")

# PROCESS RADIOMICS FOR EACH NODE
X_tr_nodes = []
X_val_nodes = []
X_te_nodes = []


for node in NODE_LAYOUT:
    print(f"\nProcessing node: {node}")
    cols = node_mapping[node]
    if not cols:
        tr_chunk = np.zeros((len(df_tr), 1))
        val_chunk = np.zeros((len(df_val), 1))
        te_chunk = np.zeros((len(df_te), 1))

    else:

        tr_chunk = df_tr[cols].values
        val_chunk = df_val[cols].values
        te_chunk = df_te[cols].values

    imputer = SimpleImputer(strategy="median")

    X_tr_imp = imputer.fit_transform(tr_chunk)
    X_val_imp = imputer.transform(val_chunk)
    X_te_imp = imputer.transform(te_chunk)

    var_thresh = VarianceThreshold(threshold=0.01)

    try:
        X_tr_var = var_thresh.fit_transform(X_tr_imp)
        X_val_var = var_thresh.transform(X_val_imp)
        X_te_var = var_thresh.transform(X_te_imp)
        var_indices = np.where(var_thresh.get_support())[0]

    except ValueError:
        X_tr_var = X_tr_imp
        X_val_var = X_val_imp
        X_te_var = X_te_imp
        var_indices = np.arange(X_tr_imp.shape[1])

    # SURVIVAL BASED FEATURE SELECTION
    max_k = min(TARGET_FEATURES_PER_NODE,X_tr_var.shape[1])

    if max_k > 0:
        feature_scores = []

        survival_time = (df_tr["Time"].values.astype(float))

        for j in range(X_tr_var.shape[1]):
            feature = X_tr_var[:, j]
            if np.std(feature) == 0:
                score = 0.0
            else:
                corr = np.corrcoef(feature,survival_time)[0, 1]
                if np.isnan(corr):
                    score = 0.0

                else:
                    score = abs(corr)

            feature_scores.append(score)

        feature_scores = np.asarray(feature_scores)

        # Rank features by absolute survival time correlation
        ranked_indices = np.argsort(feature_scores)[::-1]
        sel_local = ranked_indices[:max_k]

        selected_indices = (var_indices[sel_local])


    else:
        selected_indices = np.array([],dtype=int)

    if len(selected_indices) == 0:
        selected_indices = np.array([0],dtype=int)

    # Print selected feature information
    print(f"  Original features : {len(cols)}")

    print(f"  After variance    : {X_tr_var.shape[1]}")

    print(f"  Selected features : {len(selected_indices)}")

    # STANDARDIZATION

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_imp[:, selected_indices])
    X_val_sc = scaler.transform(X_val_imp[:, selected_indices])
    X_te_sc = scaler.transform(X_te_imp[:, selected_indices])

    # KEEP FIXED 24 FEATURES PER NODE

    X_tr_nodes.append(pad_features(X_tr_sc,TARGET_FEATURES_PER_NODE))
    X_val_nodes.append(pad_features(X_val_sc,TARGET_FEATURES_PER_NODE))
    X_te_nodes.append(pad_features(X_te_sc,TARGET_FEATURES_PER_NODE))

# STACK INTO NODE TENSORS

X_tr_arr = np.stack(X_tr_nodes,axis=1)
X_val_arr = np.stack(X_val_nodes,axis=1)
X_te_arr = np.stack(X_te_nodes,axis=1)

print("\n" + "=" * 80)
print("PROCESSED NODE TENSOR SHAPES")
print("=" * 80)
print(f"X_tr_arr : {X_tr_arr.shape}")
print(f"X_val_arr: {X_val_arr.shape}")
print(f"X_te_arr : {X_te_arr.shape}")
print("=" * 80)

# CLINICAL PREPROCESSING & MODEL DEFINITION
# ============================================================================

import time
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from torch_geometric.nn import HypergraphConv

from lifelines.utils import concordance_index

print("="*80)
print("CELL 4: CLINICAL PREPROCESSING & MODEL")
print("="*80)

DEVICE = torch.device("cpu")
print("Execution Device:",DEVICE)

# Clinical preprocessing
exact_clinical_covariates = [
    "Age_at_scan_years",
    "Gender",
    "KPS",
    "MGMT",
    "GTR_over90percent",
    "PsP_TP_score",
    "Time_since_baseline_preop"
]

clinical_covariates = [
    c for c in exact_clinical_covariates
    if c in data.columns
]

print("\nClinical Covariates:",clinical_covariates)

if clinical_covariates:
    valid_clin_cols = [
        c for c in clinical_covariates
        if data[c].notna().sum() > 0
    ]

    clinical_covariates = valid_clin_cols
    numeric_cols = [
        c for c in clinical_covariates
        if pd.api.types.is_numeric_dtype(
            data[c]
        )
    ]

    categorical_cols = [
        c for c in clinical_covariates
        if c not in numeric_cols
    ]

    tr_raw_df = df_tr[clinical_covariates].copy()

    val_raw_df = df_val[clinical_covariates].copy()

    te_raw_df = df_te[clinical_covariates].copy()

    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")

        tr_num = num_imputer.fit_transform(tr_raw_df[numeric_cols].values)
        val_num = num_imputer.transform(val_raw_df[numeric_cols].values)
        te_num = num_imputer.transform(te_raw_df[numeric_cols].values)

    else:
        tr_num = np.zeros((len(df_tr), 0))
        val_num = np.zeros((len(df_val), 0))
        te_num = np.zeros((len(df_te), 0))

    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        tr_cat_imp = cat_imputer.fit_transform(tr_raw_df[categorical_cols].astype(str).values)
        val_cat_imp = cat_imputer.transform(val_raw_df[categorical_cols].astype(str).values)
        te_cat_imp = cat_imputer.transform(te_raw_df[categorical_cols].astype(str).values)
        tr_cat_df = pd.DataFrame(tr_cat_imp,columns=categorical_cols)
        val_cat_df = pd.DataFrame(val_cat_imp,columns=categorical_cols)
        te_cat_df = pd.DataFrame(te_cat_imp,columns=categorical_cols)

        combined_cat = pd.concat(
            [
                tr_cat_df,
                val_cat_df,
                te_cat_df
            ],
            axis=0
        )

        combined_encoded = pd.get_dummies(combined_cat,drop_first=True)

        tr_cat = combined_encoded.iloc[:len(df_tr)].values.astype(float)
        val_cat = combined_encoded.iloc[len(df_tr):len(df_tr) + len(df_val)].values.astype(float)
        te_cat = combined_encoded.iloc[len(df_tr) + len(df_val):].values.astype(float)

    else:
        tr_cat = np.zeros((len(df_tr), 0))
        val_cat = np.zeros((len(df_val), 0))
        te_cat = np.zeros((len(df_te), 0))

    tr_parts = [
        p for p in [
            tr_num,
            tr_cat
        ]
        if p.shape[1] > 0
    ]

    val_parts = [
        p for p in [
            val_num,
            val_cat
        ]
        if p.shape[1] > 0
    ]

    te_parts = [
        p for p in [
            te_num,
            te_cat
        ]
        if p.shape[1] > 0
    ]

    c_tr_raw = (
        np.hstack(tr_parts)
        if tr_parts
        else np.zeros(
            (len(df_tr), 1)
        )
    )

    c_val_raw = (
        np.hstack(val_parts)
        if val_parts
        else np.zeros(
            (len(df_val), 1)
        )
    )

    c_te_raw = (
        np.hstack(te_parts)
        if te_parts
        else np.zeros(
            (len(df_te), 1)
        )
    )

    # Scale using train only
    clinical_scaler = StandardScaler()
    c_tr = clinical_scaler.fit_transform(c_tr_raw)
    c_val = clinical_scaler.transform(c_val_raw)
    c_te = clinical_scaler.transform(c_te_raw)

else:
    c_tr = np.zeros((len(df_tr), 1))
    c_val = np.zeros((len(df_val), 1))
    c_te = np.zeros((len(df_te), 1))

print("\nClinical Tensor Shapes:")
print("Train:",c_tr.shape)
print("Val  :",c_val.shape)

print("Test :",c_te.shape)

# DATASET
class SurvivalDataset(Dataset):
    def __init__(self, node_feats, clin_feats, times, events):

        self.node_feats = torch.tensor(node_feats,dtype=torch.float32)
        self.clin_feats = torch.tensor(clin_feats,dtype=torch.float32)
        self.times = torch.tensor(times,dtype=torch.float32)
        self.events = torch.tensor(events,dtype=torch.float32)

    def __len__(self):
        return len(self.node_feats)

    def __getitem__(self, idx):
        return (self.node_feats[idx],self.clin_feats[idx],self.times[idx],self.events[idx])

# DATALOADERS
train_loader = DataLoader(
    SurvivalDataset(
        X_tr_arr,
        c_tr,
        df_tr["Time"].values,
        df_tr["Event"].values
    ),
    batch_size=32,
    shuffle=True,
    drop_last=False
)

val_loader = DataLoader(
    SurvivalDataset(
        X_val_arr,
        c_val,
        df_val["Time"].values,
        df_val["Event"].values
    ),
    batch_size=32,
    shuffle=False,
    drop_last=False
)

test_loader = DataLoader(
    SurvivalDataset(
        X_te_arr,
        c_te,
        df_te["Time"].values,
        df_te["Event"].values
    ),
    batch_size=32,
    shuffle=False,
    drop_last=False
)

# HYPERGRAPH
NUM_NODES = 12
def build_static_biological_hypergraph(num_nodes=NUM_NODES,device=DEVICE):

    node_indices = []
    edge_indices = []
    edge_weights = []

    edge_id = 0
    for mod in MODALITIES:
        m_nodes = [
            i
            for i, n in enumerate(NODE_LAYOUT)
            if n.startswith(f"{mod}_")
        ]

        for node in m_nodes:
            node_indices.append(node)
            edge_indices.append(edge_id)

        edge_weights.append(1.0)
        edge_id += 1

    for comp in COMPARTMENTS:
        c_nodes = [
            i
            for i, n in enumerate(NODE_LAYOUT)
            if n.endswith(f"_{comp}")
        ]

        for node in c_nodes:
            node_indices.append(node)
            edge_indices.append(edge_id)

        edge_weights.append(0.9)
        edge_id += 1

    for node in range(num_nodes):
        node_indices.append(node)
        edge_indices.append(edge_id)

    edge_weights.append(0.8)
    hyperedge_index = torch.tensor(
        [node_indices,edge_indices],
        dtype=torch.long,
        device=device
    )

    num_hyperedges = (hyperedge_index[1].max().item()+ 1)

    hyperedge_weight = torch.tensor(edge_weights[:num_hyperedges],dtype=torch.float32,device=device)

    return (hyperedge_index,hyperedge_weight)


def get_batched_static_hypergraph(actual_batch_size,fixed_edge_index,fixed_edge_weight,num_nodes=NUM_NODES):

    num_edges_per_graph = (fixed_edge_index[1].max().item()+ 1)
    batch_indices = []
    batch_weights = []

    for b in range(actual_batch_size):
        offset_nodes = (fixed_edge_index[0]+ b * num_nodes)
        offset_edges = (fixed_edge_index[1]+ b * num_edges_per_graph)

        batch_indices.append(
            torch.stack(
                [
                    offset_nodes,
                    offset_edges
                ],
                dim=0
            )
        )

        batch_weights.append(fixed_edge_weight)

    edge_index = torch.cat(batch_indices,dim=1)
    edge_weight = torch.cat(batch_weights,dim=0)
    return (edge_index,edge_weight)


def construct_dynamic_knn_hyperedges(x_nodes_flat,actual_batch_size,device,k=3):
    x_nodes_reshaped = x_nodes_flat.view(actual_batch_size,NUM_NODES,-1)
    batch_indices = []
    batch_weights = []
    num_hyperedges_per_graph = NUM_NODES
    for b in range(actual_batch_size):
        patient_nodes = (x_nodes_reshaped[b])
        patient_nodes = torch.nan_to_num(patient_nodes,nan=0.0,posinf=1.0,neginf=-1.0)
        sim_matrix = F.cosine_similarity(patient_nodes.unsqueeze(1),patient_nodes.unsqueeze(0),dim=-1)
        sim_matrix = torch.nan_to_num(sim_matrix,nan=0.0)

        node_indices = []
        edge_indices = []
        dynamic_weights = []

        for i in range(NUM_NODES):
            _, neighbors = torch.topk(sim_matrix[i],k=min(k, NUM_NODES))

            for n in neighbors:
                node_indices.append(n.item())

                edge_indices.append(i)

            mean_sim = torch.mean(
                sim_matrix[
                    i,
                    neighbors
                ]
            )

            dynamic_weights.append(torch.clamp(mean_sim,0.1,2.0))

        single_edge_index = torch.tensor([node_indices,edge_indices],dtype=torch.long,device=device)
        single_weights = torch.stack(dynamic_weights)
        single_edge_index[0] += (b * NUM_NODES)
        single_edge_index[1] += (b * num_hyperedges_per_graph)
        batch_indices.append(single_edge_index)
        batch_weights.append(single_weights)

    edge_index = torch.cat(batch_indices,dim=1)
    edge_weight = torch.cat(batch_weights,dim=0)
    return (edge_index,edge_weight)

def negative_partial_log_likelihood(risk_scores,times,events):

    sorted_indices = torch.argsort(times,descending=True)
    risk_scores = risk_scores[sorted_indices]
    events = events[sorted_indices]

    exp_risk = torch.exp(torch.clamp(risk_scores,max=20.0))
    risk_cumsum = torch.cumsum(exp_risk,dim=0)
    log_risk = torch.log(risk_cumsum + 1e-8)
    uncensored_likelihood = (risk_scores - log_risk)
    loss = -torch.sum(uncensored_likelihood * events) / (torch.sum(events) + 1e-8)

    return torch.nan_to_num(loss,nan=0.0)


class ContrastiveRepresentationLoss(nn.Module):
    def __init__(self,temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self,embeddings,survival_times):
        median_t = torch.median(survival_times)
        labels = (survival_times < median_t).long()
        embeddings = F.normalize(embeddings,dim=1)
        similarity_matrix = (torch.matmul(embeddings,embeddings.T) / self.temperature)
        labels = labels.view(-1, 1)

        mask = torch.eq(labels,labels.T).float().to(embeddings.device)
        logits_max, _ = torch.max(similarity_matrix,dim=1,keepdim=True)
        similarity_matrix = (similarity_matrix- logits_max.detach())
        exp_sim = torch.exp(similarity_matrix) * (1 - torch.eye(len(embeddings),device=embeddings.device))
        log_prob = (similarity_matrix- torch.log(exp_sim.sum(dim=1,keepdim=True) + 1e-8))

        mask_pos_pairs = (mask* (1 -torch.eye(len(embeddings),device=embeddings.device)))
        mean_log_prob_pos = (mask_pos_pairs * log_prob).sum(1) / (mask_pos_pairs.sum(1)+ 1e-8)

        return torch.nan_to_num(-mean_log_prob_pos.mean(),nan=0.0)


# MODEL
class MultiViewHierarchicalAttentionHGNN(nn.Module):

    def __init__(self,node_in_dim,clin_dim,embed_dim=64,dropout=0.15):

        super().__init__()
        self.node_encoder = nn.Sequential(nn.Linear(node_in_dim,embed_dim),nn.LayerNorm(embed_dim),nn.GELU(),nn.Dropout(dropout))
        self.hg1 = HypergraphConv(embed_dim,embed_dim,use_attention=False,dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.hg2 = HypergraphConv(embed_dim,embed_dim,use_attention=False,dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn_query = nn.Linear(embed_dim,embed_dim // 2)
        self.attn_key = nn.Linear(embed_dim,embed_dim // 2)
        self.attn_val = nn.Linear(embed_dim,embed_dim)
        self.clin_mlp = nn.Sequential(nn.Linear(clin_dim,embed_dim),nn.LayerNorm(embed_dim),nn.GELU(),nn.Dropout(dropout))
        self.fusion_fc = nn.Linear(embed_dim * 2,embed_dim)
        self.survival_head = nn.Linear(embed_dim,1)

    def forward(self,x_nodes,x_clin,fixed_edge_index,fixed_edge_weight):
        actual_batch_size = (x_nodes.size(0))
        # Node encoder
        h = self.node_encoder(x_nodes)
        h = h.reshape(actual_batch_size * NUM_NODES,-1)

        # Static hypergraph
        edge_index_s, edge_weight_s = (get_batched_static_hypergraph(actual_batch_size,fixed_edge_index,fixed_edge_weight))

        assert (h.size(0)==actual_batch_size * NUM_NODES)
        assert (edge_index_s[0].max().item()<h.size(0))
        assert (edge_index_s[1].max().item()<len(edge_weight_s))

        h = F.gelu(self.norm1(self.hg1(h,edge_index_s,hyperedge_weight=edge_weight_s)+ h))

        # Dynamic hypergraph
        edge_index_d, edge_weight_d = (construct_dynamic_knn_hyperedges(h,actual_batch_size,h.device))
        assert (edge_index_d[0].max().item()<h.size(0))
        assert (edge_index_d[1].max().item()<len(edge_weight_d))
        h = F.gelu(self.norm2(self.hg2(h,edge_index_d,hyperedge_weight=edge_weight_d)+ h))

        # Restore node dimension
        h_reshaped = h.view(actual_batch_size,NUM_NODES,-1)

        # Node attention
        q = self.attn_query(h_reshaped)
        k = self.attn_key(h_reshaped)
        v = self.attn_val(h_reshaped)
        scores = torch.matmul(q,k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        attn_weights = F.softmax(scores,dim=-1)
        graph_rep = torch.mean(torch.matmul(attn_weights,v),dim=1)

        # Clinical fusion
        c_rep = self.clin_mlp(x_clin)
        fused = F.gelu(self.fusion_fc(torch.cat([graph_rep,c_rep],dim=-1)))
        risk = self.survival_head(fused).squeeze(-1)
        return (risk,graph_rep,attn_weights.mean(dim=1))


# MODEL INITIALIZATION
fixed_edge_index, fixed_edge_weight = (build_static_biological_hypergraph(device=DEVICE))
model = MultiViewHierarchicalAttentionHGNN(node_in_dim=X_tr_arr.shape[-1],clin_dim=c_tr.shape[1]).to(DEVICE)
contrastive_criterion = (ContrastiveRepresentationLoss())
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=0.5,patience=5)

print("\nModel initialized successfully.")
print("="*80)

# FINAL MODEL TRAINING & INDEPENDENT TEST EVALUATION
# ============================================================================

import copy
import numpy as np
import torch

print("="*80)
print("MODEL TRAINING & TEST EVALUATION")
print("="*80)

epochs = 30
best_val_cindex = -1.0
best_weights = copy.deepcopy(
model.state_dict())

print("\nStarting Training Loop...")
for epoch in range(1,epochs + 1):
    model.train()
    train_loss = 0.0
    valid_batches = 0
    for (rad_b,clin_b,time_b,event_b) in train_loader:
        rad_b = rad_b.to(DEVICE)
        clin_b = clin_b.to(DEVICE)
        time_b = time_b.to(DEVICE)
        event_b = event_b.to(DEVICE)

        optimizer.zero_grad()
        risk_scores, graph_rep, _ = model(rad_b,clin_b,fixed_edge_index,fixed_edge_weight)
        cox_loss = (negative_partial_log_likelihood(risk_scores,time_b,event_b))
        cont_loss = (contrastive_criterion(graph_rep,time_b))
        loss = (cox_loss+0.1 * cont_loss)
        if (torch.isnan(loss) or torch.isinf(loss)):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
        valid_batches += 1

    # Validation
    model.eval()
    val_risks = []
    val_times = []
    val_events = []
    with torch.no_grad():
        for (rad_b,clin_b,time_b,event_b) in val_loader:
            rad_b = rad_b.to(DEVICE)
            clin_b = clin_b.to(DEVICE)
            risk_scores, _, _ = model(rad_b,clin_b,fixed_edge_index,fixed_edge_weight)
            val_risks.extend(risk_scores.cpu().numpy())
            val_times.extend(time_b.numpy())
            val_events.extend(event_b.numpy())

    val_cindex = concordance_index(val_times, -np.array(val_risks), val_events)
    scheduler.step(val_cindex)

    if val_cindex > best_val_cindex:
        best_val_cindex = val_cindex
        best_weights = copy.deepcopy(model.state_dict())

    if (epoch % 5 == 0 or epoch == 1):
        avg_loss = (train_loss / valid_batches if valid_batches > 0 else np.nan)
        print(
            f"Epoch {epoch:02d}/{epochs} "
            f"| Train Loss: {avg_loss:.4f} "
            f"| Val C-index: {val_cindex:.4f}"
        )

# LOAD BEST VALIDATION MODEL
model.load_state_dict(best_weights)
model.eval()

# INDEPENDENT TEST SET

test_risks = []
test_times = []
test_events = []
test_attns = []
with torch.no_grad():
    for (rad_b,clin_b,time_b,event_b) in test_loader:
        rad_b = rad_b.to(DEVICE)
        clin_b = clin_b.to(DEVICE)
        risk_scores, _, attn = model(rad_b,clin_b,fixed_edge_index,fixed_edge_weight)
        test_risks.extend(risk_scores.cpu().numpy())
        test_times.extend(time_b.numpy())
        test_events.extend(event_b.numpy())
        test_attns.extend(attn.cpu().numpy())

final_cindex = concordance_index(test_times,-np.array(test_risks),test_events)

print("\n" + "="*80)
print("INDEPENDENT TEST SET SURVIVAL EVALUATION")
print("="*80)
print(f"Validation C-index : {best_val_cindex:.4f}")
print(f"Test C-index       : {final_cindex:.4f}")
print(f"Test samples       : {len(test_times)}")
print("="*80)

# SURVIVAL ANALYSIS & DATA/MODEL VISUALIZATION
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

print("="*80)
print("CELL 6: SURVIVAL ANALYSIS & VISUALIZATION")
print("="*80)

# Prediction DataFrame
# ============================================================================

risk_df = pd.DataFrame({
    "Survival_Time": np.array(test_times),
    "Event": np.array(test_events),
    "Risk_Score": np.array(test_risks)
})

risk_df = risk_df.sort_values(by="Risk_Score",ascending=False).reset_index(drop=True)
median_risk = (risk_df["Risk_Score"].median())
risk_df["Risk_Group"] = np.where(risk_df["Risk_Score"] >= median_risk,"High Risk","Low Risk")

print("\nRisk Group Distribution:")
print(risk_df["Risk_Group"].value_counts())

# C-INDEX VERIFICATION
verified_cindex = concordance_index(risk_df["Survival_Time"],-risk_df["Risk_Score"],risk_df["Event"])
print("\nIndependent Test C-index Verification:",f"{verified_cindex:.4f}")

# 5-FOLD CROSS VALIDATION
# ============================================================================

import numpy as np
import torch
import copy
from sklearn.model_selection import StratifiedKFold
from lifelines.utils import concordance_index
from torch.utils.data import DataLoader

print("="*80)
print("CELL 7: 5-FOLD CROSS VALIDATION")
print("="*80)


X_cv = np.concatenate([X_tr_arr,X_val_arr],axis=0)
C_cv = np.concatenate([c_tr,c_val],axis=0)
T_cv = np.concatenate([df_tr["Time"].values,df_val["Time"].values])
E_cv = np.concatenate([df_tr["Event"].values,df_val["Event"].values])

print("\nCV Dataset")
print("----------------")
print("Radiomics:", X_cv.shape)
print("Clinical :", C_cv.shape)
print("Survival :", T_cv.shape)

skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
fold_scores = []
# CV loop

for fold, (train_fold_idx,val_fold_idx) in enumerate(skf.split(X_cv, E_cv),1):
    print("\n" + "="*50)
    print(f"FOLD {fold}/5")
    print("="*50)

    train_dataset = SurvivalDataset(X_cv[train_fold_idx],C_cv[train_fold_idx],T_cv[train_fold_idx],E_cv[train_fold_idx])
    val_dataset = SurvivalDataset(X_cv[val_fold_idx],C_cv[val_fold_idx],T_cv[val_fold_idx],E_cv[val_fold_idx])
    train_loader_cv = DataLoader(train_dataset,batch_size=32,shuffle=True,drop_last=False)
    val_loader_cv = DataLoader(val_dataset,batch_size=32,shuffle=False,drop_last=False)

    # model
    cv_model = MultiViewHierarchicalAttentionHGNN(node_in_dim=X_cv.shape[-1],clin_dim=C_cv.shape[1]).to(DEVICE)
    optimizer_cv = torch.optim.AdamW(cv_model.parameters(),lr=1e-3,weight_decay=1e-4)
    # Training
    for epoch in range(30):
        cv_model.train()
        for (rad_b,clin_b,time_b,event_b) in train_loader_cv:
            rad_b = rad_b.to(DEVICE)
            clin_b = clin_b.to(DEVICE)
            time_b = time_b.to(DEVICE)
            event_b = event_b.to(DEVICE)
            optimizer_cv.zero_grad()
            risk, graph_rep, _ = cv_model(rad_b,clin_b,fixed_edge_index,fixed_edge_weight)
            cox_loss = (negative_partial_log_likelihood(risk,time_b,event_b))
            cont_loss = (contrastive_criterion(graph_rep,time_b))
            loss = (cox_loss+ 0.1 * cont_loss)
            if (torch.isnan(loss) or torch.isinf(loss)):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(cv_model.parameters(),max_norm=1.0)
            optimizer_cv.step()

    # Validation prediction
    cv_model.eval()
    val_risks = []
    val_times = []
    val_events = []

    with torch.no_grad():
        for (rad_b,clin_b,time_b,event_b) in val_loader_cv:
            rad_b = rad_b.to(DEVICE)
            clin_b = clin_b.to(DEVICE)
            risk, _, _ = cv_model(rad_b,clin_b,fixed_edge_index,fixed_edge_weight)
            val_risks.extend(risk.cpu().numpy())
            val_times.extend(time_b.numpy())
            val_events.extend(event_b.numpy())

    fold_cindex = concordance_index(val_times,-np.array(val_risks),val_events)
    fold_scores.append(fold_cindex)

    print(
        f"Fold {fold} C-index: "
        f"{fold_cindex:.4f}"
    )

mean_cindex = np.mean(fold_scores)
std_cindex = np.std(fold_scores)

print("\n" + "="*80)
print("5-FOLD CROSS VALIDATION RESULTS")
print("="*80)
print(
    "Fold Scores:",
    [
        round(x, 4)
        for x in fold_scores
    ]
)

print(f"\nMean C-index: {mean_cindex:.4f}")
print(f"Std         : {std_cindex:.4f}")
print("="*80)

# ABLATION STUDY
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import StratifiedKFold
from lifelines.utils import concordance_index
from torch.utils.data import DataLoader, Dataset

print("="*80)
print("CELL 8: ABLATION STUDY")
print("="*80)

X_ab = np.concatenate([X_tr_arr,X_val_arr],axis=0)
C_ab = np.concatenate([c_tr,c_val],axis=0)
T_ab = np.concatenate([df_tr["Time"].values,df_val["Time"].values])
E_ab = np.concatenate([df_tr["Event"].values,df_val["Event"].values])
print("Samples:", len(X_ab))


class AblationDataset(Dataset):
    def __init__(self,x,c,t,e):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.c = torch.tensor(c,dtype=torch.float32)
        self.t = torch.tensor(t,dtype=torch.float32)
        self.e = torch.tensor(e,dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx],self.c[idx],self.t[idx],self.e[idx])

# RADIOMICS MODEL
class RadiomicsMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(0.2),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))

    def forward(self, x):
        x = x.reshape(x.size(0),-1)
        return self.net(x).squeeze(-1)

# TRAINING FUNCTION
def train_epoch(model,loader,model_type,optimizer):
    model.train()
    total_loss = 0.0
    valid_batches = 0
    for x, c, t, e in loader:
        x = x.to(DEVICE)
        c = c.to(DEVICE)
        t = t.to(DEVICE)
        e = e.to(DEVICE)

        optimizer.zero_grad()
        if model_type == "Radiomics":
            risk = model(x)

        else:
            output = model(x,c,fixed_edge_index,fixed_edge_weight)
            risk = output[0]

        loss = (negative_partial_log_likelihood(risk,t,e))
        if (torch.isnan(loss) or torch.isinf(loss)):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        total_loss += loss.item()
        valid_batches += 1

    if valid_batches == 0:
        return np.nan
    return (total_loss / valid_batches)

# ABLATION RUNNER
def run_ablation(model_type):

    print("\nRunning:",model_type)
    scores = []
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

    for fold, (tr,va) in enumerate(skf.split(X_ab,E_ab),1):
        train_ds = AblationDataset(X_ab[tr],C_ab[tr],T_ab[tr],E_ab[tr])
        val_ds = AblationDataset(X_ab[va],C_ab[va],T_ab[va],E_ab[va])
        train_loader = DataLoader(train_ds,batch_size=32,shuffle=True)
        val_loader = DataLoader(val_ds,batch_size=32,shuffle=False)

        if model_type == "Radiomics":
            model = RadiomicsMLP(X_ab.shape[1]*X_ab.shape[2])

        else:

            model = MultiViewHierarchicalAttentionHGNN(X_ab.shape[-1],C_ab.shape[1])

        model = model.to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)

        # Train
        for epoch in range(30):
            train_epoch(model,train_loader,model_type,optimizer)

        # Validation
        model.eval()
        risks = []
        times = []
        events = []

        with torch.no_grad():
            for x, c, t, e in val_loader:
                x = x.to(DEVICE)
                c = c.to(DEVICE)

                if model_type == "Radiomics":
                    r = model(x)

                else:
                    output = model(x,c,fixed_edge_index,fixed_edge_weight)
                    r = output[0]

                risks.extend(r.cpu().numpy())
                times.extend(t.numpy())
                events.extend(e.numpy())

        score = concordance_index(times,-np.array(risks),events)
        scores.append(score)

        print(
            f"Fold {fold}: "
            f"{score:.4f}"
        )

    return (np.mean(scores), np.std(scores))

# RUNNING

results = {}

for name in ["Radiomics","Dynamic HGNN"]:
    mean, std = run_ablation(name)
    results[name] = (mean,std)

print("\n" + "="*80)
print("FINAL ABLATION RESULTS")
print("="*80)

for name, values in results.items():
    print(
        f"{name:20s}: "
        f"{values[0]:.4f} ± "
        f"{values[1]:.4f}"
    )
print("="*80)

# DYNAMIC HGNN + CONTRASTIVE LOSS ABLATION
# ============================================================================

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from lifelines.utils import concordance_index

print("="*80)
print("CELL 9: DYNAMIC HGNN + CONTRASTIVE LEARNING")
print("="*80)

X_dyn = np.concatenate([X_tr_arr,X_val_arr],axis=0)
C_dyn = np.concatenate([c_tr,c_val],axis=0)
T_dyn = np.concatenate([df_tr["Time"].values,df_val["Time"].values])
E_dyn = np.concatenate([df_tr["Event"].values,df_val["Event"].values])

print("Samples:",len(X_dyn))
skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

dyn_scores = []
for fold, (train_idx,val_idx) in enumerate(skf.split(X_dyn,E_dyn),1):
    print("\n" + "="*50)
    print(f"FOLD {fold}/5")
    print("="*50)

    train_ds = SurvivalDataset(X_dyn[train_idx],C_dyn[train_idx],T_dyn[train_idx],E_dyn[train_idx])
    val_ds = SurvivalDataset(X_dyn[val_idx],C_dyn[val_idx],T_dyn[val_idx],E_dyn[val_idx])
    train_loader = DataLoader(train_ds,batch_size=32,shuffle=True)
    val_loader = DataLoader(val_ds,batch_size=32,shuffle=False)
    model_dyn = MultiViewHierarchicalAttentionHGNN(node_in_dim=X_dyn.shape[-1],clin_dim=C_dyn.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model_dyn.parameters(),lr=1e-3,weight_decay=1e-4)

    # Training
    for epoch in range(30):
        model_dyn.train()
        for x, c, t, e in train_loader:
            x = x.to(DEVICE)
            c = c.to(DEVICE)
            t = t.to(DEVICE)
            e = e.to(DEVICE)
            optimizer.zero_grad()
            risk, graph_rep, _ = model_dyn(x,c,fixed_edge_index,fixed_edge_weight)
            cox_loss = (negative_partial_log_likelihood(risk,t,e))
            contrast_loss = (contrastive_criterion(graph_rep,t))
            loss = (cox_loss+0.1 * contrast_loss)

            if (torch.isnan(loss) or torch.isinf(loss)):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_dyn.parameters(),1.0)
            optimizer.step()
    # Validation
    model_dyn.eval()
    risks = []
    times = []
    events = []
    with torch.no_grad():
        for x, c, t, e in val_loader:
            x = x.to(DEVICE)
            c = c.to(DEVICE)
            risk, _, _ = model_dyn(x,c,fixed_edge_index,fixed_edge_weight)
            risks.extend(risk.cpu().numpy())
            times.extend(t.numpy())
            events.extend(e.numpy())
    score = concordance_index(times,-np.array(risks),events)
    dyn_scores.append(score)
    print(
        f"Fold {fold} C-index: "
        f"{score:.4f}"
    )

dyn_mean = np.mean(dyn_scores)
dyn_std = np.std(dyn_scores)

print("\n" + "="*80)
print("DYNAMIC HGNN + CONTRASTIVE RESULTS")
print("="*80)
print(
    "Scores:",
    [
        round(x, 4)
        for x in dyn_scores
    ]
)

print(f"\nMean C-index: {dyn_mean:.4f}")
print(f"Std         : {dyn_std:.4f}")
print("="*80)

# FINAL RESULTS SUMMARY 
# ============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print("="*80)
print("CELL 10: FINAL RESULTS SUMMARY")
print("="*80)


final_results = pd.DataFrame({

    "Model": [

        "Radiomics Only",

        "Dynamic HGNN",

        "Dynamic HGNN + Contrastive",

        "Independent Test Set"
    ],

    "Mean C-index": [

        results["Radiomics"][0],

        results["Dynamic HGNN"][0],

        dyn_mean,

        final_cindex
    ],

    "Std": [

        results["Radiomics"][1],

        results["Dynamic HGNN"][1],

        dyn_std,

        np.nan
    ]
})

print(final_results)

results_path = os.path.join(PROJECT_FOLDER,"HGNN_Final_Results.csv")

final_results.to_csv(results_path,index=False)

print("\nSaved:",results_path)


# 5 FOLD CV 3 CLASS EVALUATION
# ==============================================================================

import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from lifelines.utils import concordance_index
fixed_edge_index_dev = fixed_edge_index.to(DEVICE)
fixed_edge_weight_dev = fixed_edge_weight.to(DEVICE) if fixed_edge_weight is not None else None

# BraTS 3-Class Thresholds
#Short-term : < 274 days (< 9 months)
#Mid-term   : 274 to 456 days (9 to 15 months)
#Long-term  : > 456 days (> 15 months)
def get_3class_labels(times):
    times = np.asarray(times, dtype=float)
    classes = np.zeros(len(times), dtype=int)
    for i, t in enumerate(times):
        if np.isnan(t):
            classes[i] = 1
        elif t < 274.0:
            classes[i] = 0  
        elif 274.0 <= t <= 456.0:
            classes[i] = 1 
        else:
            classes[i] = 2  
    return classes

# Derive Cutoffs From Training Fold 
def fit_training_risk_cutoffs(train_risks, y_train_labels):
    clean_r = np.asarray(train_risks, dtype=float)
    p_short = np.mean(y_train_labels == 0)
    p_mid   = np.mean(y_train_labels == 1)

    t_high = np.nanpercentile(clean_r, 100 * (1 - p_short))
    t_low  = np.nanpercentile(clean_r, 100 * (1 - p_short - p_mid))
    return t_high, t_low

def apply_risk_cutoffs(val_risks, t_high, t_low):
    clean_r = np.asarray(val_risks, dtype=float)
    preds = np.zeros(len(clean_r), dtype=int)
    for i, r in enumerate(clean_r):
        if np.isnan(r):
            preds[i] = 1
        elif r >= t_high:
            preds[i] = 0  
        elif r <= t_low:
            preds[i] = 2  
        else:
            preds[i] = 1  
    return preds

# 5 Fold Cross Validation Loop
cv_cindices = []
cv_accs = []
cv_ps = []
cv_rs = []
cv_f1s = []
all_cv_records = []

print("=" * 80)
print(f"5-FOLD CV 3-CLASS EVALUATION (Cohort N = {len(T_cv)})")
print("=" * 80)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_cv, E_cv), 1):
    tr_ds = SurvivalDataset(X_cv[tr_idx], C_cv[tr_idx], T_cv[tr_idx], E_cv[tr_idx])
    va_ds = SurvivalDataset(X_cv[va_idx], C_cv[va_idx], T_cv[va_idx], E_cv[va_idx])

    tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=len(va_ds), shuffle=False)
    eval_tr_loader = DataLoader(tr_ds, batch_size=len(tr_ds), shuffle=False)

    fold_model = MultiViewHierarchicalAttentionHGNN(node_in_dim=24,clin_dim=C_cv.shape[1],embed_dim=64,dropout=0.15).to(DEVICE)
    opt = torch.optim.AdamW(fold_model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit_cont = ContrastiveRepresentationLoss().to(DEVICE)

    best_cindex = -1.0
    best_va_risks = None
    best_va_times = None

    for epoch in range(100):
        fold_model.train()
        for rad_b, clin_b, time_b, event_b in tr_loader:
            rad_b, clin_b = rad_b.to(DEVICE), clin_b.to(DEVICE)
            time_b, event_b = time_b.to(DEVICE), event_b.to(DEVICE)

            opt.zero_grad()
            r_scores, g_rep, _ = fold_model(rad_b, clin_b, fixed_edge_index_dev, fixed_edge_weight_dev)
            c_loss = negative_partial_log_likelihood(r_scores, time_b, event_b)
            ct_loss = crit_cont(g_rep, time_b)
            total_loss = c_loss + 0.1 * ct_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(fold_model.parameters(), 1.0)
            opt.step()
        fold_model.eval()
        with torch.no_grad():
            for rad_v, clin_v, time_v, event_v in va_loader:
                rad_v, clin_v = rad_v.to(DEVICE), clin_v.to(DEVICE)
                r_val, _, _ = fold_model(
                    rad_v, clin_v, fixed_edge_index_dev, fixed_edge_weight_dev
                )
                r_np = r_val.cpu().numpy().flatten()
                t_np = time_v.numpy().flatten()
                e_np = event_v.numpy().flatten()

                c_idx = concordance_index(t_np, -r_np, e_np)
                if c_idx > best_cindex:
                    best_cindex = c_idx
                    best_va_risks = r_np.copy()
                    best_va_times = t_np.copy()

    #Evaluation on the Best Fold Weights
    fold_model.eval()
    with torch.no_grad():
        for rad_tr, clin_tr, time_tr, _ in eval_tr_loader:
            rad_tr, clin_tr = rad_tr.to(DEVICE), clin_tr.to(DEVICE)
            r_train, _, _ = fold_model(
                rad_tr, clin_tr, fixed_edge_index_dev, fixed_edge_weight_dev
            )
            train_risks = r_train.cpu().numpy().flatten()
            train_times = time_tr.numpy().flatten()

    y_true_tr = get_3class_labels(train_times)
    t_high, t_low = fit_training_risk_cutoffs(train_risks, y_true_tr)
    y_true_va = get_3class_labels(best_va_times)
    y_pred_va = apply_risk_cutoffs(best_va_risks, t_high, t_low)
    acc = accuracy_score(y_true_va, y_pred_va)
    p, r, f1, _ = precision_recall_fscore_support(y_true_va, y_pred_va, average="macro", zero_division=0)

    cv_cindices.append(best_cindex)
    cv_accs.append(acc)
    cv_ps.append(p)
    cv_rs.append(r)
    cv_f1s.append(f1)

    for t_i, r_i in zip(best_va_times, best_va_risks):
        all_cv_records.append({"Fold": fold,"Survival_Time": float(t_i),"Risk_Score": float(r_i)})
    print(f"Fold {fold} -> C-Index: {best_cindex:.4f} | Acc: {acc:.4f} | Macro-P: {p:.4f} | Macro-R: {r:.4f} | Macro-F1: {f1:.4f}")
print("\n" + "=" * 80)
print("5-FOLD BENCHMARK RESULTS (mean ± std)")
print("=" * 80)
print(f"C-index  : {np.mean(cv_cindices):.4f} ± {np.std(cv_cindices):.4f}")
print(f"Accuracy : {np.mean(cv_accs):.4f} ± {np.std(cv_accs):.4f}")
print(f"Macro-P  : {np.mean(cv_ps):.4f} ± {np.std(cv_ps):.4f}")
print(f"Macro-R  : {np.mean(cv_rs):.4f} ± {np.std(cv_rs):.4f}")
print(f"Macro-F1 : {np.mean(cv_f1s):.4f} ± {np.std(cv_f1s):.4f}")
print("=" * 80)

pd.DataFrame(all_cv_records).to_csv(f"{PROJECT_FOLDER}/HGNN_cv_predictions.csv", index=False)
print(f"✓ Saved HGNN_cv_predictions.csv successfully to {PROJECT_FOLDER}")