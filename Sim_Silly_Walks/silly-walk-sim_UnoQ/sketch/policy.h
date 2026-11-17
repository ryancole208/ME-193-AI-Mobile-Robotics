// policy.h - run the trained actor on the STM32.
//
//   float obs[7] = {wx, wy, wz, sin(yaw), cos(yaw), prevActionL, prevActionR};
//   float act[2];
//   policyRun(obs, act);          // act = raw policy output (NOT clipped)
//
// Same math as the PyTorch actor: normalize -> Linear -> ELU -> Linear -> ELU -> Linear.
// Weights come from policy_weights.h (made by tools/make_policy_header.py).
#pragma once
#include <math.h>
#include "policy_weights.h"

inline float policyElu(float x) { return x > 0.0f ? x : expm1f(x); }

// out[o] = b[o] + sum_i W[o][i] * in[i]   (W is row-major [nOut][nIn], like nn.Linear.weight)
inline void policyLinear(const float* W, const float* b, const float* in, float* out, int nIn, int nOut) {
  for (int o = 0; o < nOut; o++) {
    float s = b[o];
    const float* row = W + o * nIn;
    for (int i = 0; i < nIn; i++) s += row[i] * in[i];
    out[o] = s;
  }
}

inline void policyRun(const float obs[POLICY_OBS], float act[POLICY_ACT]) {
  float x[POLICY_OBS], h1[POLICY_HID], h2[POLICY_HID];
  for (int i = 0; i < POLICY_OBS; i++)
    x[i] = (obs[i] - POLICY_OBS_MEAN[i]) / (POLICY_OBS_STD[i] + POLICY_NORM_EPS);
  policyLinear(POLICY_W0, POLICY_B0, x, h1, POLICY_OBS, POLICY_HID);
  for (int i = 0; i < POLICY_HID; i++) h1[i] = policyElu(h1[i]);
  policyLinear(POLICY_W1, POLICY_B1, h1, h2, POLICY_HID, POLICY_HID);
  for (int i = 0; i < POLICY_HID; i++) h2[i] = policyElu(h2[i]);
  policyLinear(POLICY_W2, POLICY_B2, h2, act, POLICY_HID, POLICY_ACT);
}
