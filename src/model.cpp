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
	} catch (const c10::Error &e) {
		obs_log(LOG_ERROR, "Error loading the model: %s", e.what());
	}
}

float Model::infer(uint8_t *inputBGRA, uint32_t width, uint32_t height)
{
	// convert inputBGRA to tensor
	torch::Tensor tensor = torch::from_blob(
		inputBGRA, {1, height, width, 4}, torch::kByte);
	tensor = tensor.permute({0, 3, 1, 2});
	tensor = tensor.toType(torch::kFloat);
	tensor = tensor.div(255);

	// run the model
	torch::NoGradGuard no_grad;
	torch::Tensor output = m_module.forward({tensor}).toTensor();

	// get the output
	float confidence = output[0].item<float>();
	return confidence;
}