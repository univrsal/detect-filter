#include "model.hpp"
#include "plugin-support.h"
#include <torch/script.h>

Model::Model() {}

Model::~Model() {}

void Model::load_model(const std::string &model_path)
{
	if (model_path == m_model_path)
		return;

	m_model_path = model_path;
	m_loaded = false;

	// unload the model if it is already loaded
	try {
		m_module = torch::jit::load(model_path);
		m_module.eval();
		m_loaded = true;

		if (torch::cuda::is_available()) {
			m_module.to(torch::kCUDA);
		}
	} catch (const c10::Error &e) {
		obs_log(LOG_ERROR, "Error loading the model: %s", e.what());
	}
}

// Function to convert RGB byte array to grayscale tensor
torch::Tensor convertToGrayscale(torch::Tensor rgbTensor)
{
	// Select the first channel (R)
	torch::Tensor red_channel = rgbTensor.index({torch::indexing::Slice(), 0, torch::indexing::Slice(), torch::indexing::Slice()});

	// Select the second channel (G)
	torch::Tensor green_channel = rgbTensor.index({torch::indexing::Slice(), 1, torch::indexing::Slice(), torch::indexing::Slice()});

	// Select the third channel (B)
	torch::Tensor blue_channel = rgbTensor.index({torch::indexing::Slice(), 2, torch::indexing::Slice(), torch::indexing::Slice()});

	// Compute the grayscale tensor
	torch::Tensor grayTensor = 0.299 * red_channel + 0.587 * green_channel + 0.114 * blue_channel;

	grayTensor = grayTensor.unsqueeze(0);

	return grayTensor;
}

torch::Tensor applyLaplacianFilter(torch::Tensor grayImage)
{
	// Define the 3x3 Laplacian kernel
	torch::Tensor laplacianKernel =
		torch::tensor({{0, 1, 0}, {1, -4, 1}, {0, 1, 0}},
			      torch::kFloat32)
			.unsqueeze(0)
			.unsqueeze(0)
			.to(grayImage.device());


	// Expand dimensions of the grayscale image tensor to (1, 1, H, W) to match the input format required for convolution
	torch::Tensor grayImageExpanded = grayImage;//.squeeze(0).squeeze(0);

	obs_log(LOG_INFO, "%i, %i, %i, %i", laplacianKernel.size(0), laplacianKernel.size(1),laplacianKernel.size(2),laplacianKernel.size(3));
	obs_log(LOG_INFO, "%i, %i, %i, %i", grayImageExpanded.size(0), grayImageExpanded.size(1),grayImageExpanded.size(2),grayImageExpanded.size(3));
	// Apply 2D convolution using the Laplacian kernel
	torch::Tensor laplacianImage =
		torch::conv2d(grayImageExpanded, laplacianKernel);

	// Remove extra dimensions to get back to (H, W) shape
	laplacianImage = laplacianImage.squeeze(0).squeeze(0);

	return laplacianImage;
}

float Model::infer(uint8_t *inputBGRA, uint32_t width, uint32_t height)
{
	if (!m_loaded)
		return -1;
	// convert inputBGRA to tensor
	torch::Tensor tensor = torch::from_blob(
		inputBGRA, {1, width, height, 4}, torch::kByte);
	tensor = tensor.permute({0, 3, 1, 2});
	tensor = tensor.toType(torch::kFloat);
	tensor = tensor.div(255);

	if (torch::cuda::is_available()) {
		tensor = tensor.to(torch::kCUDA);
	}

	auto grayTensor = convertToGrayscale(tensor);

	auto laplacianImage = applyLaplacianFilter(grayTensor);

	// run the model
	torch::NoGradGuard no_grad;
	torch::Tensor output = m_module.forward({laplacianImage}).toTensor();

	if (torch::cuda::is_available())
		torch::cuda::synchronize();

	output = output.contiguous().to(torch::kCPU);

	// get the output
	auto softmax = torch::nn::functional::softmax(output, 1);
	float score = softmax[0][0].item<float>();
	return score;
}
