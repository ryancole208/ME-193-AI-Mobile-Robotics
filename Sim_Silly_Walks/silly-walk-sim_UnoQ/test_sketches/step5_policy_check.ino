// =====================================================================================
// Silly Walk Sim - STEP 5: CHECK THE C POLICY ON THE STM32 (motors stay OFF)
//
// Runs policyRun() on the 20 fixed test observations from policy_test_vectors.h and prints
// the actions, then times the network. tools/check_policy_c.py compares the printed actions
// with the Python reference (tools/policy_test_vectors.json).
// Everything is re-printed every 5 s so a late-connecting monitor still sees it.
// =====================================================================================

#include <Arduino_RouterBridge.h>
#include "motors.h"
#include "policy.h"
#include "policy_test_vectors.h"

float results[N_POLICY_TESTS][POLICY_ACT];
float usPerRun = 0;

void setup() {
  Monitor.begin();
  stopArms();
  for (int t = 0; t < N_POLICY_TESTS; t++) policyRun(POLICY_TEST_OBS[t], results[t]);

  const int N = 1000;              // time 1000 runs of the network
  float a[POLICY_ACT];
  unsigned long t0 = micros();
  for (int i = 0; i < N; i++) policyRun(POLICY_TEST_OBS[i % N_POLICY_TESTS], a);
  usPerRun = (micros() - t0) / (float)N;
}

void loop() {
  char line[96];
  Monitor.println("# POLICY CHECK BEGIN");
  for (int t = 0; t < N_POLICY_TESTS; t++) {
    snprintf(line, sizeof(line), "V %d %.7f %.7f", t, results[t][0], results[t][1]);
    Monitor.println(line);
  }
  snprintf(line, sizeof(line), "# TIME %.1f us per policy step (budget: 20000 us)", usPerRun);
  Monitor.println(line);
  Monitor.println("# POLICY CHECK END");
  delay(5000);
}
