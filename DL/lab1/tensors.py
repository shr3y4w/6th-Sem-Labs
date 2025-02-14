import torch
import numpy as np

# 1. Reshaping, viewing, stacking, squeezing, and unsqueezing

tensor1 = torch.arange(1, 10)
print("Original tensor:\n", tensor1)

# Reshaping
reshaped_tensor = tensor1.view(3, 3)
print("\nReshaped tensor:\n", reshaped_tensor)

# Stacking
tensor2 = torch.arange(10, 19).view(3, 3)
stacked_tensor = torch.stack((reshaped_tensor, tensor2), dim=0)  # Stack along a new dimension. new shape (stacked structures, row, col)
print("\nStacked tensor:\n", stacked_tensor)

# Squeezing
tensor3 = torch.tensor([[[1, 2, 3]]])
squeezed_tensor = tensor3.squeeze()  # Remove dimensions of size 1
print("\nSqueezed tensor:\n", squeezed_tensor)

# Unsqueezing
unsqueezed_tensor = tensor1.unsqueeze(0)  # Add a new dimension at position 0
print("\nUnsqueezed tensor:\n", unsqueezed_tensor)


# 2. Using torch.permute()
# Permute changes the order of dimensions
tensor4 = torch.randn(2, 3, 4)  # A random 3D tensor
permuted_tensor = tensor4.permute(2, 0, 1)  # Change dimension order: (2, 3, 4) -> (4, 2, 3)
print("\nOriginal shape:", tensor4.shape)
print(tensor4)
print(permuted_tensor)
print("Permuted shape:", permuted_tensor.shape)

# 3. Indexing in tensors

tensor5 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("\nOriginal tensor:\n", tensor5)

print("\nElement at row 1, column 2:", tensor5[1, 2])
print("Entire row 2:", tensor5[1, :])
print("Entire column 3:", tensor5[:, 2])

# 4. Converting numpy arrays to tensors and back
numpy_array = np.array([1.0, 2.0, 3.0])
tensor6 = torch.from_numpy(numpy_array)
print("\nTensor from numpy array:", tensor6)

# Convert tensor back to numpy array
numpy_back = tensor6.numpy()
print("Numpy array from tensor:", numpy_back)

# 5. Create a random tensor with shape (7, 7)

tensor7 = torch.rand(7, 7)
print("\nRandom tensor (7x7):\n", tensor7)

# 6. Perform matrix multiplication

tensor8 = torch.rand(1, 7)
result = torch.matmul(tensor7, tensor8.T)  # Transpose tensor8 for compatibility!!
print("\nResult of matrix multiplication:\n", result)

# 7. Create two random tensors of shape (2, 3) and send them to GPU

if torch.cuda.is_available():  # Check if GPU is available
    tensor9 = torch.rand(2, 3).cuda()
    tensor10 = torch.rand(2, 3).cuda()
    print("\nRandom tensors on GPU:")
    print("Tensor 9:\n", tensor9)
    print("Tensor 10:\n", tensor10)

# 8. Perform matrix multiplication on GPU tensors

    #adjust shapes if needed
    tensor11 = tensor10.T  # Transpose tensor10 to make dimensions compatible
    result_gpu = torch.matmul(tensor9, tensor11)
    print("\nResult of matrix multiplication on GPU:\n", result_gpu)

    # 9. Find the maximum and minimum values
    max_val = torch.max(result_gpu)
    min_val = torch.min(result_gpu)
    print("\nMax value on GPU tensor:", max_val.item())
    print("Min value on GPU tensor:", min_val.item())

    # 10. Find the indices of max and min values
    max_idx = torch.argmax(result_gpu)
    min_idx = torch.argmin(result_gpu)
    print("\nIndex of max value:", max_idx.item())
    print("Index of min value:", min_idx.item())

# 11. Create a random tensor with shape (1, 1, 1, 10) and remove dimensions
torch.manual_seed(7)  # Set seed for reproducibility
tensor12 = torch.rand(1, 1, 1, 10)  # Shape (1, 1, 1, 10)
squeezed_tensor12 = tensor12.squeeze()  # Remove dimensions of size 1
print("\nOriginal tensor:\n", tensor12)
print("Shape of original tensor:", tensor12.shape)
print("Squeezed tensor:\n", squeezed_tensor12)
print("Shape of squeezed tensor:", squeezed_tensor12.shape)

