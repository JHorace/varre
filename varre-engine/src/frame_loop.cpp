/**
 * @file frame_loop.cpp
 * @brief Frame-loop primitive implementation.
 */
#include "varre/engine/frame_loop.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

#include "varre/engine/engine.hpp"
#include "varre/engine/swapchain.hpp"

namespace varre::engine {
  namespace {
    /**
     * @brief Append semaphore only when valid and not already present.
     * @param semaphores Destination semaphore list.
     * @param semaphore Semaphore to append.
     */
    void append_unique_semaphore(std::vector<vk::Semaphore> *semaphores, const vk::Semaphore semaphore) {
      if (semaphore == VK_NULL_HANDLE) {
        return;
      }
      const bool exists = std::ranges::any_of(*semaphores, [&](const vk::Semaphore value) { return value == semaphore; });
      if (!exists) {
        semaphores->push_back(semaphore);
      }
    }
  } // namespace

  FrameLoop::FrameLoop(
      const vk::raii::Device *device,
      const vk::Queue graphics_queue,
      const vk::Queue present_queue,
      std::vector<FrameSyncPrimitives> &&frames,
      std::vector<vk::Fence> &&image_in_flight_fences
  )
    : device_(device),
      graphics_queue_(graphics_queue),
      present_queue_(present_queue),
      frames_(std::move(frames)),
      image_in_flight_fences_(std::move(image_in_flight_fences)) {
  }

  FrameLoop FrameLoop::create(const EngineContext &engine, const SwapchainContext &swapchain, const FrameLoopCreateInfo &info) {
    const vk::raii::Device &device = engine.device();
    const SurfaceQueueTopology &queue_topology = swapchain.queue_topology();
    std::uint32_t frame_count = info.frame_count == 0U ? swapchain.max_frames_in_flight() : info.frame_count;
    frame_count = std::clamp(frame_count, 1U, swapchain.image_count());

    const vk::SemaphoreCreateInfo semaphore_create_info{};
    const vk::FenceCreateInfo fence_create_info = vk::FenceCreateInfo{}.setFlags(vk::FenceCreateFlagBits::eSignaled);

    std::vector<FrameSyncPrimitives> frames;
    frames.reserve(frame_count);
    for (std::uint32_t index = 0; index < frame_count; ++index) {
      FrameSyncPrimitives frame{};
      frame.image_available = vk::raii::Semaphore(device, semaphore_create_info);
      frame.render_finished = vk::raii::Semaphore(device, semaphore_create_info);
      frame.in_flight = vk::raii::Fence(device, fence_create_info);
      frames.push_back(std::move(frame));
    }

    std::vector<vk::Fence> image_in_flight_fences(swapchain.image_count(), VK_NULL_HANDLE);
    return FrameLoop{
        &device,
        queue_topology.graphics_queue,
        queue_topology.present_queue,
        std::move(frames),
        std::move(image_in_flight_fences),
    };
  }

  AcquiredFrame FrameLoop::acquire_next_image(const SwapchainContext &swapchain, const std::uint64_t timeout_ns) {
    if (device_ == nullptr || frames_.empty()) {
      throw std::runtime_error("FrameLoop is not initialized.");
    }
    if (swapchain.image_count() == 0U) {
      throw std::runtime_error("Swapchain has no images.");
    }
    if (image_in_flight_fences_.size() != swapchain.image_count()) {
      throw std::runtime_error("Swapchain image count changed; call FrameLoop::reset_for_swapchain before acquiring.");
    }

    AcquiredFrame result{
        .status = FrameAcquireStatus::kSuccess,
        .frame_index = current_frame_index_,
        .image_index = 0U,
    };
    frame_submitted_ = false;
    render_finished_signaled_ = false;

    FrameSyncPrimitives &frame = frames_[current_frame_index_];
    const std::array in_flight_fence{*frame.in_flight};
    const vk::Result in_flight_wait_result = device_->waitForFences(in_flight_fence, VK_TRUE, timeout_ns);
    if (in_flight_wait_result != vk::Result::eSuccess) {
      throw std::runtime_error("Timed out while waiting for the current frame fence.");
    }

    try {
      const vk::ResultValue<std::uint32_t> acquire_result = swapchain.swapchain().acquireNextImage(timeout_ns, *frame.image_available, vk::Fence{});
      result.image_index = acquire_result.value;
      if (acquire_result.result == vk::Result::eSuboptimalKHR) {
        result.status = FrameAcquireStatus::kSuboptimal;
        swapchain_recreation_required_ = true;
      } else {
        result.status = FrameAcquireStatus::kSuccess;
      }
    } catch (const vk::OutOfDateKHRError &) {
      frame_acquired_ = false;
      swapchain_recreation_required_ = true;
      result.status = FrameAcquireStatus::kOutOfDate;
      return result;
    }

    if (image_in_flight_fences_[result.image_index] != VK_NULL_HANDLE) {
      const std::array image_fence{image_in_flight_fences_[result.image_index]};
      const vk::Result image_wait_result = device_->waitForFences(image_fence, VK_TRUE, timeout_ns);
      if (image_wait_result != vk::Result::eSuccess) {
        throw std::runtime_error("Timed out while waiting for a prior in-flight image fence.");
      }
    }
    image_in_flight_fences_[result.image_index] = *frame.in_flight;

    device_->resetFences(in_flight_fence);
    frame_acquired_ = true;
    return result;
  }

  void FrameLoop::submit_graphics_batch(const GraphicsSubmitBatch &batch) {
    if (!frame_acquired_) {
      throw std::runtime_error("submit_graphics_batch called before acquire_next_image.");
    }

    std::vector<vk::Semaphore> wait_semaphores;
    std::vector<vk::PipelineStageFlags> wait_stage_masks;
    wait_semaphores.reserve(batch.waits.size() + (batch.wait_for_swapchain_image ? 1U : 0U));
    wait_stage_masks.reserve(wait_semaphores.capacity());
    for (const SubmitSemaphoreWait &wait : batch.waits) {
      if (wait.semaphore == VK_NULL_HANDLE) {
        continue;
      }
      wait_semaphores.push_back(wait.semaphore);
      wait_stage_masks.push_back(wait.stage_mask);
    }

    const FrameSyncPrimitives &frame = frames_[current_frame_index_];
    if (batch.wait_for_swapchain_image) {
      wait_semaphores.push_back(*frame.image_available);
      wait_stage_masks.push_back(batch.swapchain_image_wait_stage);
    }

    std::vector<vk::Semaphore> signal_semaphores = batch.signals;
    if (batch.signal_render_finished) {
      append_unique_semaphore(&signal_semaphores, *frame.render_finished);
    }

    const vk::SubmitInfo submit_info = vk::SubmitInfo{}
        .setWaitSemaphores(wait_semaphores)
        .setWaitDstStageMask(wait_stage_masks)
        .setCommandBuffers(batch.command_buffers)
        .setSignalSemaphores(signal_semaphores);
    graphics_queue_.submit(submit_info, *frame.in_flight);

    frame_submitted_ = true;
    render_finished_signaled_ = batch.signal_render_finished;
  }

  void FrameLoop::submit_graphics(const vk::CommandBuffer command_buffer, const vk::PipelineStageFlags wait_stage_mask) {
    GraphicsSubmitBatch batch{
      .waits = {},
      .command_buffers = {command_buffer},
      .signals = {},
      .wait_for_swapchain_image = true,
      .swapchain_image_wait_stage = wait_stage_mask,
      .signal_render_finished = true,
    };
    submit_graphics_batch(batch);
  }

  FramePresentStatus FrameLoop::present(const SwapchainContext &swapchain, const FramePresentRequest &request) {
    if (!frame_acquired_) {
      throw std::runtime_error("present called before acquire_next_image.");
    }
    if (!frame_submitted_) {
      throw std::runtime_error("present called before submitting graphics work.");
    }
    if (request.image_index >= swapchain.image_count()) {
      throw std::runtime_error("present called with an out-of-range swapchain image index.");
    }

    const FrameSyncPrimitives &frame = frames_[current_frame_index_];
    const vk::SwapchainKHR swapchain_handle = *swapchain.swapchain();
    std::vector<vk::Semaphore> present_waits = request.wait_semaphores;
    if (request.include_render_finished) {
      if (!render_finished_signaled_) {
        throw std::runtime_error("present requested render-finished wait, but submit did not signal it for this frame.");
      }
      append_unique_semaphore(&present_waits, *frame.render_finished);
    }
    const vk::PresentInfoKHR present_info = vk::PresentInfoKHR{}
        .setWaitSemaphores(present_waits)
        .setSwapchains(swapchain_handle)
        .setImageIndices(request.image_index);

    FramePresentStatus status = FramePresentStatus::kSuccess;
    try {
      const vk::Result result = present_queue_.presentKHR(present_info);
      if (result == vk::Result::eSuboptimalKHR) {
        status = FramePresentStatus::kSuboptimal;
        swapchain_recreation_required_ = true;
      } else if (result != vk::Result::eSuccess) {
        throw std::runtime_error("Queue present failed with unexpected Vulkan result.");
      }
    } catch (const vk::OutOfDateKHRError &) {
      swapchain_recreation_required_ = true;
      status = FramePresentStatus::kOutOfDate;
    }

    frame_acquired_ = false;
    frame_submitted_ = false;
    render_finished_signaled_ = false;
    current_frame_index_ = (current_frame_index_ + 1U) % static_cast<std::uint32_t>(frames_.size());
    return status;
  }

  FramePresentStatus FrameLoop::present(const SwapchainContext &swapchain, const std::uint32_t image_index) {
    return present(
        swapchain,
        FramePresentRequest{
          .image_index = image_index,
          .wait_semaphores = {},
          .include_render_finished = true,
        }
    );
  }

  bool FrameLoop::swapchain_recreation_required() const noexcept { return swapchain_recreation_required_; }

  void FrameLoop::recreate_swapchain(SwapchainContext *swapchain, const SwapchainCreateInfo &recreate_info) {
    if (swapchain == nullptr) {
      throw std::runtime_error("FrameLoop::recreate_swapchain requires a valid SwapchainContext pointer.");
    }
    wait_idle();
    *swapchain = swapchain->recreate(recreate_info);
    notify_swapchain_recreated(*swapchain);
  }

  void FrameLoop::recreate_swapchain(SwapchainContext *swapchain) {
    if (swapchain == nullptr) {
      throw std::runtime_error("FrameLoop::recreate_swapchain requires a valid SwapchainContext pointer.");
    }
    wait_idle();
    *swapchain = swapchain->recreate();
    notify_swapchain_recreated(*swapchain);
  }

  void FrameLoop::notify_swapchain_recreated(const SwapchainContext &swapchain) {
    present_queue_ = swapchain.queue_topology().present_queue;
    image_in_flight_fences_.assign(swapchain.image_count(), VK_NULL_HANDLE);
    frame_acquired_ = false;
    frame_submitted_ = false;
    render_finished_signaled_ = false;
    swapchain_recreation_required_ = false;
    if (current_frame_index_ >= frames_.size()) {
      current_frame_index_ = 0U;
    }
  }

  void FrameLoop::reset_for_swapchain(const SwapchainContext &swapchain) {
    notify_swapchain_recreated(swapchain);
  }

  void FrameLoop::wait_idle() const {
    if (device_ != nullptr) {
      device_->waitIdle();
    }
  }

  std::uint32_t FrameLoop::current_frame_index() const noexcept { return current_frame_index_; }

  std::uint32_t FrameLoop::frame_count() const noexcept { return static_cast<std::uint32_t>(frames_.size()); }

  const FrameSyncPrimitives &FrameLoop::current_sync() const noexcept { return frames_[current_frame_index_]; }
} // namespace varre::engine
