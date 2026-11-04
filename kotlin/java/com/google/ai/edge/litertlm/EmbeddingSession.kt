/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.ai.edge.litertlm

import java.util.concurrent.atomic.AtomicBoolean

/**
 * The outputs of one [EmbeddingSession.stepEmbedding].
 *
 * @property hidden The hidden states of the step: the output of the model's last layer, before the
 *   logits.
 * @property logits The logits of the step, or null if they were not requested.
 */
class EmbeddingStep(val hidden: FloatArray, val logits: FloatArray?)

/**
 * Runs the engine's model step by step with rows of input embeddings, for models that take input
 * embeddings and output their hidden states, e.g. a speech generation loop that feeds back
 * embeddings computed from the hidden states. Every row takes one position of the session's
 * context. Created by [Engine.createEmbeddingSession].
 *
 * The session shares the engine's loaded model with the engine's other sessions and conversations,
 * and holds its own context: switching between them copies the model state (e.g. KV cache). The
 * calls are blocking; do not call them concurrently with each other or with calls of another
 * session or conversation of the same engine.
 *
 * @param handle The pointer to the underlying native session object.
 */
class EmbeddingSession internal constructor(private val handle: Long) : AutoCloseable {
  private val _isAlive = AtomicBoolean(true)

  /** Whether the session is alive and ready to be used. */
  val isAlive: Boolean
    get() = _isAlive.get()

  /** The number of positions of the session's context that are filled. */
  val currentStep: Int
    get() {
      checkIsAlive()
      return LiteRtLmJni.nativeSessionGetCurrentStep(handle)
    }

  /**
   * Returns the input embedding row of the text token [tokenId], from the model's embedder.
   *
   * @throws IllegalStateException if the session is not alive.
   * @throws LiteRtLmJniException if the model has no embedder.
   */
  fun embedToken(tokenId: Int): FloatArray {
    checkIsAlive()
    return LiteRtLmJni.nativeSessionGetTokenEmbedding(handle, tokenId)
  }

  /**
   * The number of floats in a row of input embeddings: the width of the model's embedder rows,
   * which is the width of the model's embeddings input.
   *
   * @throws LiteRtLmJniException if the model has no embedder.
   */
  val embeddingSize: Int by lazy { embedToken(0).size }

  /**
   * Prefills [count] rows of input embeddings, the first `count * embeddingSize` floats of [rows],
   * at the current step.
   *
   * @return The hidden states of the last row.
   * @throws IllegalStateException if the session is not alive.
   * @throws IllegalArgumentException if [count] is not positive or [rows] holds fewer rows.
   * @throws LiteRtLmJniException if the model does not take input embeddings or output hidden
   *   states, or if the rows do not fit in the context.
   */
  fun prefillEmbeddings(rows: FloatArray, count: Int): FloatArray {
    checkIsAlive()
    require(count > 0) { "The number of rows must be positive, got $count." }
    val numFloats = count * embeddingSize
    require(rows.size >= numFloats) {
      "$count rows of $embeddingSize floats need $numFloats floats, got ${rows.size}."
    }
    return LiteRtLmJni.nativeSessionPrefillEmbeddings(handle, rows, numFloats)
  }

  /**
   * Runs one step with the input embedding row [row] at the current step.
   *
   * @param wantLogits Whether to compute the logits of the step. A step without logits skips the
   *   model's output head when the model has a prefill signature of length 1.
   * @throws IllegalStateException if the session is not alive.
   * @throws LiteRtLmJniException if the model does not take input embeddings or output hidden
   *   states, or if the context is full.
   */
  fun stepEmbedding(row: FloatArray, wantLogits: Boolean): EmbeddingStep {
    checkIsAlive()
    val outputs = LiteRtLmJni.nativeSessionDecodeEmbedding(handle, row, wantLogits)
    return EmbeddingStep(hidden = outputs[0]!!, logits = outputs[1])
  }

  /**
   * Starts a new sequence: rewinds the session to position 0 with a zeroed model state.
   *
   * @throws IllegalStateException if the session is not alive.
   */
  fun reset() {
    checkIsAlive()
    LiteRtLmJni.nativeSessionRewindToStep(handle, 0)
  }

  /**
   * Closes the session and releases the native session's resources.
   *
   * @throws IllegalStateException if the session has already been closed.
   */
  override fun close() {
    if (_isAlive.compareAndSet(true, false)) {
      LiteRtLmJni.nativeDeleteSession(handle)
    } else {
      throw IllegalStateException("EmbeddingSession is closed already.")
    }
  }

  private fun checkIsAlive() {
    check(isAlive) { "EmbeddingSession is not alive." }
  }
}
