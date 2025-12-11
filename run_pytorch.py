"""
PyTorch training script for PGGCN model on Mobley dataset (cd-set1).

This script:
1. Reads the Mobley dataset info.csv
2. Filters for cd-set1 dataset
3. Constructs host-ligand complexes from PDB files
4. Trains the PGGCN model with 80/20 train/test split
5. Evaluates the model on test set
"""

import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
import pandas as pd
import numpy as np
import sys
import os
import random
from sklearn.model_selection import train_test_split

sys.path.append('/home/ali/PycharmProjects/GBNN')

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from PGGCN.models.dcFeaturizer import atom_features as get_atom_features
from PGGCN.models.layers_pytorch import PGGCNModel


class PhysicsInformedLoss(nn.Module):
    """
    Custom loss function that combines MSE with a sign penalty.

    The loss penalizes when model_var has a different sign than (pb_host_Etot - pb_guest_Etot).
    This enforces physics-based constraints on the model's predictions.

    Loss = MSE(prediction, target) + sign_penalty_weight * sign_loss
    """

    def __init__(self, sign_penalty_weight=1.0):
        super(PhysicsInformedLoss, self).__init__()
        self.sign_penalty_weight = sign_penalty_weight
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets, model_vars, physics_info):
        """
        Args:
            predictions: Final model predictions [batch_size, 1]
            targets: Ground truth values [batch_size, 1]
            model_vars: Model predictions before physics fusion [batch_size, 1]
            physics_info: Physics-based info [batch_size, 2] where
                         [:, 0] = pb_guest_Etot
                         [:, 1] = pb_host_Etot

        Returns:
            Combined loss value
        """
        # Standard MSE loss
        mse_loss = self.mse(predictions, targets)

        # Calculate physics-based sign
        # pb_host_Etot - pb_guest_Etot
        physics_diff = physics_info[:, 1] - physics_info[:, 0]  # [batch_size]
        physics_sign = torch.sign(physics_diff)  # [batch_size]

        # Model prediction sign
        model_sign = torch.sign(model_vars.squeeze(-1))  # [batch_size]

        # Sign mismatch penalty
        # If signs match: sign_product = 1 (positive)
        # If signs mismatch: sign_product = -1 (negative)
        sign_product = physics_sign * model_sign  # [batch_size]

        # Penalize when sign_product < 0 (signs don't match)
        # Use ReLU on negative sign_product to create penalty
        sign_penalty = torch.mean(torch.relu(-sign_product))

        # Combined loss
        total_loss = mse_loss + self.sign_penalty_weight * sign_penalty

        return total_loss, mse_loss, sign_penalty


def featurize(molecule, info):
    """
    Featurize a molecule with additional info array.

    Args:
        molecule: RDKit molecule object
        info: List of additional information (e.g., [pb_guest_Etot, pb_host_Etot])

    Returns:
        numpy array of atom features
    """
    atom_features = []
    for atom in molecule.GetAtoms():
        # Base features from DeepChem (32 features)
        base_feat = get_atom_features(atom)
        new_feature = base_feat.tolist()

        # Add position information
        position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
        new_feature += [atom.GetMass(), atom.GetAtomicNum(), atom.GetFormalCharge()]
        new_feature += [position.x, position.y, position.z]

        # At this point we have 32 + 3 + 3 = 38 features

        # Add neighbor indices (up to 2 neighbors)
        neighbors = atom.GetNeighbors()[:2]
        for neighbor in neighbors:
            neighbor_idx = neighbor.GetIdx()
            new_feature += [float(neighbor_idx)]

        # Pad if less than 2 neighbors
        for i in range(2 - len(neighbors)):
            new_feature += [0.0]

        # Now we have 38 + 2 = 40 features

        # Concatenate info array
        full_feature = new_feature + info
        atom_features.append(full_feature)

    return np.array(atom_features)


def load_cd_set1_data(info_csv_path, pdb_dir):
    """
    Load cd-set1 dataset from info.csv and PDB files.

    Args:
        info_csv_path: Path to info.csv
        pdb_dir: Directory containing PDB files

    Returns:
        X: List of feature tensors
        y: List of target values (Ex _G_(kcal/mol))
        info: Dataset information
    """
    # Read CSV
    df = pd.read_csv(info_csv_path)

    # Filter for cd-set1
    df_cd_set1 = df[df['Dataset Name'] == 'cd-set1'].copy()

    print(f"Found {len(df_cd_set1)} entries in cd-set1 dataset")

    # Load host molecule (acd)
    host_path = os.path.join(pdb_dir, 'host-acd.pdb')
    host_mol = Chem.MolFromPDBFile(host_path)

    if host_mol is None:
        raise ValueError(f"Failed to load host molecule from {host_path}")

    print(f"Loaded host molecule: {host_mol.GetNumAtoms()} atoms")

    X = []
    y = []
    failed = []

    # Process each entry
    for idx, row in df_cd_set1.iterrows():
        guest_name = row['Guest']
        target = row['Ex _G_(kcal/mol)']
        pb_guest_etot = row['pb_guest_Etot']
        pb_host_etot = row['pb_host_Etot']

        # Load guest molecule
        guest_path = os.path.join(pdb_dir, f'{guest_name}.pdb')

        if not os.path.exists(guest_path):
            print(f"Warning: Guest file not found: {guest_path}")
            failed.append(guest_name)
            continue

        guest_mol = Chem.MolFromPDBFile(guest_path)

        if guest_mol is None:
            print(f"Warning: Failed to load guest molecule: {guest_path}")
            failed.append(guest_name)
            continue

        # Combine host and guest
        complex_mol = Chem.CombineMols(host_mol, guest_mol)

        # Create info array
        info_array = [pb_guest_etot, pb_host_etot]

        # Featurize
        features = featurize(complex_mol, info_array)

        # Debug: Check neighbor indices
        if len(X) == 0:  # First sample
            print(f"\nDebug info for first sample ({guest_name}):")
            print(f"  Complex has {complex_mol.GetNumAtoms()} atoms")
            print(f"  Feature shape: {features.shape}")
            print(f"  Expected: [num_atoms, 30(base) + 3(mass,atomic_num,charge) + 3(xyz) + 2(neighbors) + {len(info_array)}(info)] = {38 + len(info_array)}")
            print(f"  Sample neighbor indices from first 3 atoms:")
            for atom_idx in range(min(3, features.shape[0])):
                neighbor_vals = features[atom_idx, 36:38]
                print(f"    Atom {atom_idx}: neighbors = {neighbor_vals}")
            print(f"  Sample info values from first atom: {features[0, 38:]}")

        # Convert to tensor
        features_tensor = torch.FloatTensor(features)

        X.append(features_tensor)
        y.append(target)

    print(f"Successfully loaded {len(X)} complexes")
    if failed:
        print(f"Failed to load: {failed}")

    if len(X) < 20:
        print(f"\n⚠️  WARNING: Very small dataset ({len(X)} samples)!")
        print(f"   With such few samples, results will be highly variable and may not generalize well.")
        print(f"   This is a proof-of-concept run. For reliable results, use more data.")

    return X, y, df_cd_set1


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, lr=0.001, device='cpu',
                sign_penalty_weight=1.0):
    """
    Train the PGGCN model with physics-informed loss.

    Args:
        model: PGGCNModel instance
        X_train: Training features (list of tensors)
        y_train: Training targets (list)
        X_val: Validation features (list of tensors)
        y_val: Validation targets (list)
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        sign_penalty_weight: Weight for sign penalty in loss function

    Returns:
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = PhysicsInformedLoss(sign_penalty_weight=sign_penalty_weight)

    train_losses = []
    val_losses = []
    train_mse_losses = []
    train_sign_penalties = []

    # Move data to device
    X_train = [x.to(device) for x in X_train]
    X_val = [x.to(device) for x in X_val]
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)

    print(f"\nTraining on {len(X_train)} samples, validating on {len(X_val)} samples")
    print(f"Device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Sign penalty weight: {sign_penalty_weight}")
    print(f"Epochs: {epochs}")
    print("=" * 80)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()

        # Forward pass - now returns (predictions, model_var, physics_info)
        predictions, model_var, physics_info = model(X_train)

        # Compute loss with physics-informed penalty
        train_loss, train_mse, train_sign_penalty = criterion(
            predictions, y_train_tensor, model_var, physics_info
        )

        # Backward pass
        train_loss.backward()
        optimizer.step()

        train_losses.append(train_loss.item())
        train_mse_losses.append(train_mse.item())
        train_sign_penalties.append(train_sign_penalty.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions, val_model_var, val_physics_info = model(X_val)
            val_loss, val_mse, val_sign_penalty = criterion(
                val_predictions, y_val_tensor, val_model_var, val_physics_info
            )
            val_losses.append(val_loss.item())

            # Calculate RMSE (from MSE component only)
            train_rmse = torch.sqrt(train_mse).item()
            val_rmse = torch.sqrt(val_mse).item()

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss.item():.4f} (MSE: {train_mse.item():.4f}, "
                  f"Sign: {train_sign_penalty.item():.4f}) | "
                  f"Val Loss: {val_loss.item():.4f} (RMSE: {val_rmse:.4f})")

        # Save best model
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_epoch = epoch + 1

    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

    return train_losses, val_losses


def evaluate_model(model, X_test, y_test, device='cpu'):
    """
    Evaluate the model on test set.

    Args:
        model: Trained PGGCNModel
        X_test: Test features (list of tensors)
        y_test: Test targets (list)
        device: Device to evaluate on

    Returns:
        predictions: Model predictions
        metrics: Dictionary of evaluation metrics
    """
    model.eval()

    # Move data to device
    X_test = [x.to(device) for x in X_test]
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)

    with torch.no_grad():
        predictions, model_var, physics_info = model(X_test)

        # Calculate metrics
        mse = nn.MSELoss()(predictions, y_test_tensor).item()
        rmse = np.sqrt(mse)
        mae = torch.mean(torch.abs(predictions - y_test_tensor)).item()

        # R^2 score
        ss_res = torch.sum((y_test_tensor - predictions) ** 2).item()
        ss_tot = torch.sum((y_test_tensor - torch.mean(y_test_tensor)) ** 2).item()
        r2 = 1 - (ss_res / ss_tot)

        # Calculate sign accuracy
        physics_diff = physics_info[:, 1] - physics_info[:, 0]
        physics_sign = torch.sign(physics_diff)
        model_sign = torch.sign(model_var.squeeze(-1))
        sign_matches = (physics_sign == model_sign).float()
        sign_accuracy = torch.mean(sign_matches).item()

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'sign_accuracy': sign_accuracy
    }

    return predictions.cpu().numpy(), metrics


def main():
    print("=" * 80)
    print("PyTorch PGGCN Training on Mobley cd-set1 Dataset")
    print("=" * 80)

    # Paths
    info_csv_path = '/home/ali/PycharmProjects/GBNN/Datasets/Mobley/info.csv'
    pdb_dir = '/home/ali/PycharmProjects/GBNN/Datasets/Mobley/cd-set1/pdb'

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Load data
    print("\n" + "-" * 80)
    print("Loading Data")
    print("-" * 80)
    X, y, df_info = load_cd_set1_data(info_csv_path, pdb_dir)

    # Split data (80/20)
    print("\n" + "-" * 80)
    print("Splitting Data (80% train, 20% test)")
    print("-" * 80)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Create model
    print("\n" + "-" * 80)
    print("Creating Model")
    print("-" * 80)
    # num_atom_features = 30 (base) + 3 (mass, atomic_num, charge) + 3 (x, y, z) = 36
    model = PGGCNModel(num_atom_features=36, r_out_channel=20, c_out_channel=1024)

    # Add rules (matching the TensorFlow version)
    # Features: [0:30 base, 30 mass, 31 atomic_num, 32 charge, 33:36 xyz]
    model.add_rule("sum", 0, 30)  # Sum base features
    model.add_rule("multiply", 30, 31)  # Multiply mass
    model.add_rule("distance", 33, 36)  # Distance on xyz coordinates

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Rule-based graph convolution rules: {len(model.rule_graph_conv.combination_rules)}")
    print(f"\nModel Architecture:")
    print(f"  1. RuleGraphConvLayer: {model.num_atom_features} → {20} (with custom rules)")
    print(f"  2. ConvLayer: {20} → {1024}")
    print(f"  3. Dense1: {1024} → {32}")
    print(f"  4. Dense2: {32} → {16}")
    print(f"  5. Dense3: {16} → {1} (model prediction)")
    print(f"  6. Concatenate with physics info: [1 + 2] → 3")
    print(f"  7. DenseFinal: {3} → {1} (final refined prediction)")

    # Train model
    print("\n" + "-" * 80)
    print("Training Model")
    print("-" * 80)

    # Sign penalty weight: higher values enforce stronger sign matching
    sign_penalty_weight = 1.0

    train_losses, val_losses = train_model(
        model, X_train, y_train, X_test, y_test,
        epochs=200,
        lr=0.001,
        device=device,
        sign_penalty_weight=sign_penalty_weight
    )

    # Evaluate on test set
    print("\n" + "-" * 80)
    print("Evaluating on Test Set")
    print("-" * 80)
    predictions, metrics = evaluate_model(model, X_test, y_test, device=device)

    print(f"\nTest Set Metrics:")
    print(f"  MSE:  {metrics['mse']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  Sign Accuracy: {metrics['sign_accuracy']*100:.1f}%")

    # Show some predictions
    print("\n" + "-" * 80)
    print("Sample Predictions")
    print("-" * 80)
    print(f"{'True Value':>12} | {'Prediction':>12} | {'Error':>12}")
    print("-" * 40)
    for i in range(min(10, len(y_test))):
        true_val = y_test[i]
        pred_val = predictions[i][0]
        error = pred_val - true_val
        print(f"{true_val:>12.4f} | {pred_val:>12.4f} | {error:>12.4f}")

    # Save model
    print("\n" + "-" * 80)
    print("Saving Model")
    print("-" * 80)
    model_path = '/home/ali/PycharmProjects/GBNN/pggcn_pytorch_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'metrics': metrics
    }, model_path)
    print(f"Model saved to: {model_path}")

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
