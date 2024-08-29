/*
detect-filter
Copyright (C) 2024 Alex <uni@vrsal.xyz>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>
*/
#include "filter.hpp"
#include "model.hpp"
#include "plugin-support.h"

#include <string>
#include <mutex>

#define T_(s) obs_module_text(s)
#define T_CONFIDENCE_THRESHOLD T_("confidencethreshold")
#define T_CONFIDENCE_INFO T_("model")
#define T_LOG T_("log")

#define S_MODEL_PATH "model"
#define S_CONFIDENCE_THRESHOLD "confidence_threshold"
#define S_LOG "log"

struct detect_filter_data {
	obs_source_t *source{};
	gs_texrender_t *texrender{};
	gs_stagesurf_t *stagesurface{};
	std::mutex inputBGRALock{};
	uint8_t *inputBGRA{};
	Model model{};
	bool m_log{};
	float confidenceThreshold{};
	uint32_t width{}, height{}, bpc{}; // bytes per channel
};

bool getRGBAFromStageSurface(detect_filter_data *tf, uint32_t &width,
			     uint32_t &height)
{

	if (!obs_source_enabled(tf->source)) {
		return false;
	}

	obs_source_t *target = obs_filter_get_target(tf->source);
	if (!target) {
		return false;
	}
	width = obs_source_get_base_width(target);
	height = obs_source_get_base_height(target);
	if (width == 0 || height == 0) {
		return false;
	}
	gs_texrender_reset(tf->texrender);
	if (!gs_texrender_begin(tf->texrender, width, height)) {
		return false;
	}
	struct vec4 background;
	vec4_zero(&background);
	gs_clear(GS_CLEAR_COLOR, &background, 0.0f, 0);
	gs_ortho(0.0f, static_cast<float>(width), 0.0f,
		 static_cast<float>(height), -100.0f, 100.0f);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	obs_source_video_render(target);
	gs_blend_state_pop();
	gs_texrender_end(tf->texrender);

	if (tf->stagesurface) {
		uint32_t stagesurf_width =
			gs_stagesurface_get_width(tf->stagesurface);
		uint32_t stagesurf_height =
			gs_stagesurface_get_height(tf->stagesurface);
		if (stagesurf_width != width || stagesurf_height != height) {
			gs_stagesurface_destroy(tf->stagesurface);
			tf->stagesurface = nullptr;
		}
	}
	if (!tf->stagesurface) {
		tf->stagesurface =
			gs_stagesurface_create(width, height, GS_BGRA);
	}
	gs_stage_texture(tf->stagesurface,
			 gs_texrender_get_texture(tf->texrender));
	uint8_t *video_data;
	uint32_t linesize;
	if (!gs_stagesurface_map(tf->stagesurface, &video_data, &linesize)) {
		return false;
	}
	{
		std::lock_guard<std::mutex> lock(tf->inputBGRALock);
		bfree(tf->inputBGRA);
		tf->bpc = linesize/width;
		tf->inputBGRA = (uint8_t *)bmalloc(width * height * tf->bpc);
		memcpy(tf->inputBGRA, video_data, width * height * tf->bpc);
	}
	gs_stagesurface_unmap(tf->stagesurface);
	return true;
}

static const char *df_name(void *)
{
	return "Detect filter";
}

static void df_destroy(void *data)
{
	auto filter = static_cast<detect_filter_data *>(data);

	obs_enter_graphics();
	gs_texrender_destroy(filter->texrender);
	if (filter->stagesurface) {
		gs_stagesurface_destroy(filter->stagesurface);
	}
	obs_leave_graphics();
	bfree(filter->inputBGRA);
	delete filter;
}

static void df_update(void *data, obs_data_t *s)
{
	auto filter = static_cast<detect_filter_data *>(data);

	const char *model_path = obs_data_get_string(s, S_MODEL_PATH);
	filter->model.load_model(model_path);
	filter->confidenceThreshold =
		(float)obs_data_get_double(s, S_CONFIDENCE_THRESHOLD);
}

static void *df_create(obs_data_t *settings, obs_source_t *filter)
{
	auto *mfd = new detect_filter_data;
	mfd->source = filter;
	mfd->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);

	df_update(mfd, settings);
	return mfd;
}

static obs_properties_t *df_properties(void *data)
{
	UNUSED_PARAMETER(data);
	auto *props = obs_properties_create();

	obs_properties_add_float(props, S_CONFIDENCE_THRESHOLD,
				 T_CONFIDENCE_THRESHOLD, 0.01, 1.0, 0.01);
	obs_properties_add_path(props, S_MODEL_PATH, T_CONFIDENCE_INFO,
				OBS_PATH_FILE, "Pytorch models (*.pt)",
				nullptr);

	obs_properties_add_bool(props, S_LOG, T_LOG);

	return props;
}

static void df_defaults(obs_data_t *s)
{
	obs_data_set_default_double(s, S_CONFIDENCE_THRESHOLD, 0.2);
	obs_data_set_default_string(s, S_MODEL_PATH, "");
	obs_data_set_default_bool(s, S_LOG, false);
}

static void df_video_render(void *data, gs_effect_t *effect)
{
	UNUSED_PARAMETER(effect);
	auto filter = static_cast<detect_filter_data *>(data);
	obs_source_skip_video_filter(filter->source);

	uint32_t width, height;
	if (getRGBAFromStageSurface(filter, width, height)) {
		filter->width = width;
		filter->height = height;
	} else {
		obs_log(LOG_DEBUG, "Failed to get RGBA from stage surface");
	}
}

static void df_video_tick(void *data, float)
{
	auto filter = static_cast<detect_filter_data *>(data);

	uint8_t *inputBGRA{};
	uint32_t width{}, height{};
	{
		std::lock_guard<std::mutex> lock(filter->inputBGRALock);
		if (filter->inputBGRA) {
			inputBGRA = (uint8_t *)bmalloc(filter->width *
						       filter->height *
						       filter->bpc);
			memcpy(inputBGRA, (void *)filter->inputBGRA,
			       filter->width * filter->height *
				       filter->bpc);
		}
		width = filter->width;
		height = filter->height;
	}

	if (inputBGRA) {
		float confidence =
			filter->model.infer(inputBGRA, width, height);

		obs_log(LOG_DEBUG, "confidence: %f", confidence);
		bfree(inputBGRA);
	}
}

struct obs_source_info detect_filter = {
	.id = "detect_filter",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_VIDEO,
	.get_name = df_name,
	.create = df_create,
	.destroy = df_destroy,
	.get_defaults = df_defaults,
	.get_properties = df_properties,
	.update = df_update,
	.video_tick = df_video_tick,
	.video_render = df_video_render,
};
