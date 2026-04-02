import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Make sure these imports match your exact project structure
from external.tokenhmr.tokenhmr.lib.utils.rotation_utils import rotation_6d_to_matrix
from external.tokenhmr.tokenization.models.vanilla_pose_vqvae import EncodeTokens, DecodeTokens, body_model

def save_obj(vertices, faces, filename="frankenstein_output.obj"):
    """Saves the 3D mesh to an OBJ file without needing extra libraries."""
    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            # OBJ faces are 1-indexed, so we add 1
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"Saved mesh to {filename}")

def plot_joints(joints, title="3D Pose Joints"):
    """Creates a quick 3D scatter plot of the joints using matplotlib."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # SMPL joints are (X, Y, Z). 
    xs = joints[:, 0]
    ys = joints[:, 1]
    zs = joints[:, 2]
    
    ax.scatter(xs, ys, zs, c='r', marker='o')
    
    # Set labels and keep axes equal for proper proportions
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Hack to keep 3D aspect ratio roughly square in matplotlib
    max_range = max(xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()) / 2.0
    mid_x = (xs.max()+xs.min()) * 0.5
    mid_y = (ys.max()+ys.min()) * 0.5
    mid_z = (zs.max()+zs.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.show()

def run_frankenstein_test(ckpt_path, neutral_pose_6d, token_idx_to_change, target_codebook_idx):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("Loading Encoder...")
    encoder = EncodeTokens(ckpt_path=ckpt_path).to(device).eval()
    print("Loading Decoder...")
    decoder = DecodeTokens(ckpt_path=ckpt_path).to(device).eval()
    print("Loading SMPL body model...")
    
    # Ensure body model is on the correct device
    global body_model 
    body_model = body_model.to(device)
    neutral_pose_6d = neutral_pose_6d.to(device)

    with torch.no_grad():
        neutral_pose_tokens = encoder(neutral_pose_6d)
        print("Original tokens:", neutral_pose_tokens.cpu().numpy())

        # 1. Modify the tokens as integers first
        modified_tokens_int = neutral_pose_tokens.clone()
        if modified_tokens_int.dim() == 1:
            modified_tokens_int[token_idx_to_change] = target_codebook_idx
        else:
            modified_tokens_int[0, token_idx_to_change] = target_codebook_idx
            
        print("Modified tokens (ints):", modified_tokens_int.cpu().numpy())

        # 2. Convert to One-Hot Logits
        # We need to know the codebook size. Based on your error, it is 2048.
        codebook_size = 2048 
        
        # Ensure modified_tokens_int is (1, 160)
        if modified_tokens_int.dim() == 1:
            tokens_for_onehot = modified_tokens_int.unsqueeze(0)
        else:
            tokens_for_onehot = modified_tokens_int

        # Create one-hot: shape (1, 160, 2048)
        one_hot_logits = torch.nn.functional.one_hot(tokens_for_onehot.long(), num_classes=codebook_size).float()

        # 3. Decode using the one-hot logits
        reconstructed_pose_6d = decoder(one_hot_logits)
        
        # 4. Process back to Rotmat and Mesh
        pred_rotmat = rotation_6d_to_matrix(reconstructed_pose_6d.view(-1, 6)).view(1, 21, 3, 3)
        output_mesh = body_model(body_pose=pred_rotmat)

        vertices = output_mesh.vertices.cpu().numpy()[0]
        joints = output_mesh.joints.cpu().numpy()[0]

        print("Reconstruction was successful!")
        return vertices, joints

if __name__ == '__main__':
    # 1. Setup your paths and parameters
    ckpt_path = '/home/marco/thesis-HMR/external/tokenhmr/data/checkpoints/tokenizer.pth'
    
    # FIX: Input shape must be (Batch, Num_Joints, 6D_Features)
    neutral_pose_6d = torch.zeros(1, 21, 6) 
    
    # Note: If zeros act weird, replace the line above with a real frame from your dataset!
    
    token_idx_to_change = 5  
    target_codebook_idx = 10 

    # 2. Run the test
    vertices, joints = run_frankenstein_test(ckpt_path, neutral_pose_6d, token_idx_to_change, target_codebook_idx)

    # 3. Visualize the output
    print("Plotting joints...")
    plot_joints(joints, title=f"Token {token_idx_to_change} -> ID {target_codebook_idx}")
    
    print("Saving OBJ file...")
    # body_model.faces contains the SMPL triangle definitions needed to draw the surface
    faces = body_model.faces
    save_obj(vertices, faces, filename=f"frankenstein_t{token_idx_to_change}_id{target_codebook_idx}.obj")