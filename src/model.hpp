#pragma once

#include <torch/torch.h>
#include <obs-module.h>

class Model {
	torch::jit::script::Module m_module;
	bool m_loaded = false;
	std::string m_model_path{};

public:
	Model();
	~Model();
	void load_model(const std::string &model_path);

	float infer(uint8_t *inputBGRA, uint32_t width, uint32_t height);
};
