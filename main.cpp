#include "raylib.h"
#include <cmath>
#include <vector>
#include <string>

// ---------- Basic vector helpers ----------
static float Length(const Vector2& v) { return sqrtf(v.x * v.x + v.y * v.y); }
static Vector2 Normalize(const Vector2& v) {
    float l = Length(v);
    if (l == 0) return { 0,0 };
    return { v.x / l, v.y / l };
}
static Vector2 Scale(const Vector2& v, float s) { return { v.x * s, v.y * s }; }
static Vector2 Add(const Vector2& a, const Vector2& b) { return { a.x + b.x, a.y + b.y }; }
static Vector2 Sub(const Vector2& a, const Vector2& b) { return { a.x - b.x, a.y - b.y }; }
static Vector2 Limit(const Vector2& v, float max) {
    float l = Length(v);
    if (l > max) return Scale(v, max / l);
    return v;
}

#ifndef PI
static const float PI = 3.14159265358979323846f;
#endif

// ---------- Single-agent behaviors (Task 1) ----------
Vector2 Seek(const Vector2& pos, const Vector2& target, float maxSpeed) {
    Vector2 desired = Sub(target, pos);
    desired = Normalize(desired);
    return Scale(desired, maxSpeed);
}
Vector2 Flee(const Vector2& pos, const Vector2& target, float maxSpeed) {
    Vector2 desired = Sub(pos, target);
    desired = Normalize(desired);
    return Scale(desired, maxSpeed);
}
Vector2 Pursue(const Vector2& pos, const Vector2& targetPos, const Vector2& targetVel, float maxSpeed, float predictionFactor = 0.5f) {
    Vector2 toTarget = Sub(targetPos, pos);
    float dist = Length(toTarget);
    float T = dist / (maxSpeed + 0.0001f) * predictionFactor;
    Vector2 future = Add(targetPos, Scale(targetVel, T));
    return Seek(pos, future, maxSpeed);
}
Vector2 Evade(const Vector2& pos, const Vector2& targetPos, const Vector2& targetVel, float maxSpeed, float predictionFactor = 0.5f) {
    Vector2 toTarget = Sub(targetPos, pos);
    float dist = Length(toTarget);
    float T = dist / (maxSpeed + 0.0001f) * predictionFactor;
    Vector2 future = Add(targetPos, Scale(targetVel, T));
    return Flee(pos, future, maxSpeed);
}
Vector2 Arrive(const Vector2& pos, const Vector2& target, float maxSpeed, float slowingRadius) {
    Vector2 desired = Sub(target, pos);
    float dist = Length(desired);
    if (dist < 0.001f) return { 0,0 };
    desired = Normalize(desired);
    float speed = maxSpeed * (dist / slowingRadius);
    if (speed > maxSpeed) speed = maxSpeed;
    return Scale(desired, speed);
}
Vector2 Wander(const Vector2& position, const Vector2& velocity, float maxSpeed, float& wanderAngle) {
    float circleDistance = 50;
    float circleRadius = 30;
    float angleChange = 0.5f;

    Vector2 circleCenter = Normalize(velocity);
    if (Length(circleCenter) < 0.01f) circleCenter = { 0, -1 };
    circleCenter = Scale(circleCenter, circleDistance);

    // random change
    wanderAngle += ((float)GetRandomValue(-100, 100) / 100.0f) * angleChange;

    Vector2 displacement = { cosf(wanderAngle) * circleRadius, sinf(wanderAngle) * circleRadius };
    Vector2 wanderForce = Add(circleCenter, displacement);
    return Limit(wanderForce, maxSpeed);
}

// ---------- Multi-agent system for Task2 ----------
struct Agent {
    Vector2 pos;
    Vector2 vel;
    Vector2 acc;
    float maxSpeed;
    float maxForce;
    int pathIndex;
    Color color;
};

Vector2 ArriveSteer(const Agent& a, const Vector2& target, float slowingRadius) {
    Vector2 desired = Arrive(a.pos, target, a.maxSpeed, slowingRadius);
    return Sub(desired, a.vel); // steering = desired - velocity
}

Vector2 PredictiveAvoidance(const Agent& a, const Agent& b, float lookAheadTime, float maxAvoidForce) {
    Vector2 futureA = Add(a.pos, Scale(a.vel, lookAheadTime));
    Vector2 futureB = Add(b.pos, Scale(b.vel, lookAheadTime));
    Vector2 diff = Sub(futureA, futureB);
    float dist = Length(diff);
    float combinedRadius = 24.0f;
    if (dist < combinedRadius && dist > 0.001f) {
        Vector2 away = Normalize(Sub(futureA, futureB));
        float strength = (combinedRadius - dist) / combinedRadius;
        return Scale(away, maxAvoidForce * (0.4f + 0.6f * strength));
    }
    return { 0,0 };
}

Vector2 Separation(const Agent& self, const std::vector<Agent>& agents, float separationRadius, float strength) {
    Vector2 steer = { 0,0 };
    int count = 0;
    for (const Agent& other : agents) {
        // compare by address isn't reliable for vector iteration copies, so compare positions (safe)
        if (&other == &self) {
            // fallback: skip if same pos and vel (best-effort)
            if (&other == &self) continue;
        }
        Vector2 diff = Sub(self.pos, other.pos);
        float d = Length(diff);
        if (d > 0 && d < separationRadius) {
            Vector2 away = Normalize(diff);
            float factor = (separationRadius - d) / separationRadius;
            steer = Add(steer, Scale(away, factor));
            count++;
        }
    }
    if (count > 0) steer = Scale(steer, 1.0f / (float)count);
    if (Length(steer) < 0.0001f) return { 0,0 };
    steer = Normalize(steer);
    return Scale(steer, strength);
}

Vector2 ObstacleAvoidance(const Agent& agent, const std::vector<Vector2>& obsCenters, const std::vector<float>& obsRadii, float lookAhead, float avoidStrength) {
    Vector2 heading = Normalize(agent.vel);
    if (Length(heading) < 0.01f) heading = { 0, -1 };
    Vector2 ahead = Add(agent.pos, Scale(heading, lookAhead));
    Vector2 steer = { 0,0 };
    for (size_t i = 0;i < obsCenters.size();++i) {
        Vector2 c = obsCenters[i];
        float r = obsRadii[i] + 8.0f; // buffer
        float dist = Length(Sub(ahead, c));
        if (dist < r) {
            Vector2 away = Normalize(Sub(ahead, c));
            float penetration = (r - dist);
            steer = Add(steer, Scale(away, penetration * avoidStrength));
        }
        else {
            float nowDist = Length(Sub(agent.pos, c));
            if (nowDist < r) {
                Vector2 awayNow = Normalize(Sub(agent.pos, c));
                steer = Add(steer, Scale(awayNow, (r - nowDist) * avoidStrength * 0.8f));
            }
        }
    }
    if (Length(steer) < 0.001f) return { 0,0 };
    return Limit(steer, avoidStrength);
}

Vector2 WallAvoidance(const Agent& a, float screenW, float screenH, float margin, float strength) {
    Vector2 steer = { 0,0 };
    if (a.pos.x < margin) steer.x = strength * (1.0f - (a.pos.x / margin));
    else if (a.pos.x > screenW - margin) steer.x = -strength * (1.0f - ((screenW - a.pos.x) / margin));
    if (a.pos.y < margin) steer.y = strength * (1.0f - (a.pos.y / margin));
    else if (a.pos.y > screenH - margin) steer.y = -strength * (1.0f - ((screenH - a.pos.y) / margin));
    return steer;
}

Vector2 PathFollowing(const Agent& a, const std::vector<Vector2>& path, int& outIndex, float waypointRadius) {
    if (path.empty()) return { 0,0 };
    if (outIndex >= (int)path.size()) outIndex = 0;
    Vector2 target = path[outIndex];
    float dist = Length(Sub(target, a.pos));
    if (dist < waypointRadius) {
        outIndex = (outIndex + 1) % (int)path.size();
        target = path[outIndex];
    }
    return Arrive(a.pos, target, a.maxSpeed, waypointRadius * 2.5f);
}

// ---------- Task3: Combining behaviors ----------
Vector2 PrioritySteering(const std::vector<Vector2>& forces, float epsilon = 0.001f) {
    for (const Vector2& f : forces) {
        if (Length(f) > epsilon) return f;
    }
    return { 0,0 };
}
Vector2 WeightedBlend(const std::vector<std::pair<Vector2, float>>& forces, float maxForce) {
    Vector2 total = { 0,0 };
    for (const auto& p : forces) {
        total = Add(total, Scale(p.first, p.second));
    }
    return Limit(total, maxForce);
}

// ---------- Drawing helpers ----------
void DrawAgentTriangle(const Vector2& pos, const Vector2& vel, Color color) {
    // Robust triangle draw:
    // - If velocity is near zero, pick a default heading (right)
    // - Larger triangle (so it's visible on big screens)
    // - Draw filled triangle + outline

    float heading;
    if (Length(vel) < 0.01f) {
        // default facing right (so it is visible even if agent is stationary)
        heading = 0.0f;
    }
    else {
        heading = atan2f(vel.y, vel.x);
    }

    // triangle size
    float forward = 20.0f;      // length from center to tip
    float halfBase = 12.0f;     // half of base width

    // define local points (pointing toward +X)
    Vector2 p1 = { forward, 0 };                 // tip
    Vector2 p2 = { -halfBase, halfBase * 0.7f }; // bottom-left
    Vector2 p3 = { -halfBase, -halfBase * 0.7f };// top-left

    // rotation matrix
    float s = sinf(heading), c = cosf(heading);
    auto rot = [&](const Vector2& p) {
        return Vector2{ pos.x + (p.x * c - p.y * s), pos.y + (p.x * s + p.y * c) };
        };

    Vector2 r1 = rot(p1), r2 = rot(p2), r3 = rot(p3);

    // Draw filled + outline for better visibility
    DrawTriangle(r1, r2, r3, color);
    DrawTriangleLines(r1, r2, r3, BLACK);
}

// ---------- Main ----------
int main() {
    const int screenW = 1800, screenH = 1000;
    InitWindow(screenW, screenH, "Steering Behaviors Assignment - Fixed");
    SetTargetFPS(60);

    // --- Task1 single-agent setup ---
    Agent player;
    player.pos = { 500,400 };
    // *** small non-zero initial velocity so heading is defined immediately ***
    player.vel = { 0.05f, 0.0f };
    player.acc = { 0,0 };
    player.maxSpeed = 3.0f;
    player.maxForce = 0.12f;
    float wanderAngle = 0.0f;
    Vector2 target = { 700,500 };
    int singleMode = 1; // 1=Seek 2=Flee 3=Pursue 4=Evade 5=Arrive 6=Wander
    Vector2 mousePrev = GetMousePosition();
    Vector2 mouseVel = { 0,0 };

    // --- Task2 multi-agent setup ---
    std::vector<Vector2> path = {
        {150,120}, {400,90}, {800,150}, {920,300},
        {800,520}, {520,620}, {240,500}, {100,350}
    };
    std::vector<Vector2> obsCenters = { {500,320}, {300,380}, {700,460} };
    std::vector<float> obsRadii = { 60.0f, 45.0f, 55.0f };

    std::vector<Agent> agents;
    const int AGENT_COUNT = 12;
    for (int i = 0; i < AGENT_COUNT; ++i) {
        Agent a;
        a.pos = { (float)GetRandomValue(80, screenW - 80), (float)GetRandomValue(80, screenH - 80) };
        a.vel = { (float)GetRandomValue(-50,50) / 10.0f, (float)GetRandomValue(-50,50) / 10.0f };
        a.acc = { 0,0 };
        a.maxSpeed = 2.4f + (GetRandomValue(0, 30) / 100.0f);
        a.maxForce = 0.14f;
        a.pathIndex = GetRandomValue(0, (int)path.size() - 1);
        a.color = (i % 2 == 0) ? SKYBLUE : MAROON;
        agents.push_back(a);
    }

    // Toggles & weights
    bool singleAgentMode = true; // if true show Task1 single-agent, else multi-agent Task2
    bool enablePathFollowing = true;
    bool enableSeparation = true;
    bool enablePredictiveAvoid = true;
    bool enableObstacleAvoid = true;
    bool enableWallAvoid = true;
    bool drawDebug = true;
    bool usePriority = true; // Task3: use priority blending vs weighted blending

    // --- NEW: single-agent combining toggle (Task3 demonstration) ---
    bool singleCombine = false; // press 'B' to toggle combining for single agent

    float separationRadius = 48.0f;
    float separationStrength = 0.9f;
    float predictiveLookAhead = 0.9f;
    float predictiveStrength = 0.9f;
    float obstacleLookAhead = 70.0f;
    float obstacleStrength = 1.2f;
    float wallMargin = 40.0f;
    float wallStrength = 1.6f;
    float pathWaypointRadius = 22.0f;

    // main loop
    while (!WindowShouldClose()) {
        float dt = GetFrameTime();

        // Input toggles
        if (IsKeyPressed(KEY_TAB)) singleAgentMode = !singleAgentMode;
        if (IsKeyPressed(KEY_ONE)) {
            if (singleAgentMode) singleMode = 1; else enablePathFollowing = !enablePathFollowing;
        }
        if (IsKeyPressed(KEY_TWO)) {
            if (singleAgentMode) singleMode = 2; else enableSeparation = !enableSeparation;
        }
        if (IsKeyPressed(KEY_THREE)) {
            if (singleAgentMode) singleMode = 3; else enablePredictiveAvoid = !enablePredictiveAvoid;
        }
        if (IsKeyPressed(KEY_FOUR)) {
            if (singleAgentMode) singleMode = 4; else enableObstacleAvoid = !enableObstacleAvoid;
        }
        if (IsKeyPressed(KEY_FIVE)) {
            if (singleAgentMode) singleMode = 5; else enableWallAvoid = !enableWallAvoid;
        }
        if (IsKeyPressed(KEY_SIX)) {
            if (singleAgentMode) singleMode = 6;
        }
        if (IsKeyPressed(KEY_D)) drawDebug = !drawDebug;
        if (IsKeyPressed(KEY_P)) usePriority = !usePriority; // switch combining approach
        if (IsKeyPressed(KEY_B)) singleCombine = !singleCombine; // NEW: toggle single-agent combining demo

        // Mouse target & mouse velocity estimation for pursue/evade
        target = GetMousePosition();
        Vector2 mNow = target;
        mouseVel = Sub(mNow, mousePrev);
        mousePrev = mNow;

        // ---------- Single-agent behavior (Task1) ----------
        if (singleAgentMode) {
            Vector2 steering = { 0,0 };

            // Optionally demonstrate combining (Task3) for single agent:
            if (singleCombine) {
                // Weighted blend of Wander (exploration) + Seek (goal-directed)
                Vector2 wanderF = Wander(player.pos, player.vel, player.maxSpeed, wanderAngle);
                Vector2 seekF = Seek(player.pos, target, player.maxSpeed);
                std::vector<std::pair<Vector2, float>> wforces;
                wforces.push_back({ wanderF, 0.6f }); // wander 60%
                wforces.push_back({ seekF,   1.0f }); // seek stronger
                steering = WeightedBlend(wforces, player.maxForce * 2.0f); // produce desired velocity-ish
            }
            else {
                switch (singleMode) {
                case 1: steering = Seek(player.pos, target, player.maxSpeed); break;
                case 2: steering = Flee(player.pos, target, player.maxSpeed); break;
                case 3: steering = Pursue(player.pos, target, mouseVel, player.maxSpeed, 0.8f); break;
                case 4: steering = Evade(player.pos, target, mouseVel, player.maxSpeed, 0.8f); break;
                case 5: steering = Arrive(player.pos, target, player.maxSpeed, 140.0f); break;
                case 6: steering = Wander(player.pos, player.vel, player.maxSpeed, wanderAngle); break;
                }
            }

            // steering returned is desired velocity (for Seek/Flee/Arrive/Wander). Convert to steering = desired - vel
            Vector2 steerVec = Sub(steering, player.vel);
            steerVec = Limit(steerVec, player.maxForce);

            // integrate (no explicit dt scaling here — frame-rate stable enough for demo)
            player.vel = Add(player.vel, steerVec);
            player.vel = Limit(player.vel, player.maxSpeed);
            player.pos = Add(player.pos, player.vel);

            // keep inside screen
            if (player.pos.x < 0) player.pos.x = 0;
            if (player.pos.y < 0) player.pos.y = 0;
            if (player.pos.x > screenW) player.pos.x = screenW;
            if (player.pos.y > screenH) player.pos.y = screenH;
        }
        // ---------- Multi-agent behaviors (Task2) ----------
        else {
            for (size_t i = 0; i < agents.size(); ++i) {
                Agent& a = agents[i];
                a.acc = { 0,0 };

                // compute component behaviors
                Vector2 steerPath = { 0,0 };
                if (enablePathFollowing) {
                    Vector2 desired = PathFollowing(a, path, a.pathIndex, pathWaypointRadius);
                    steerPath = Sub(desired, a.vel);
                }

                Vector2 steerSep = { 0,0 };
                if (enableSeparation) {
                    steerSep = Separation(a, agents, separationRadius, separationStrength);
                }

                Vector2 steerPredict = { 0,0 };
                if (enablePredictiveAvoid) {
                    for (size_t j = 0; j < agents.size(); ++j) {
                        if (i == j) continue;
                        Vector2 s = PredictiveAvoidance(a, agents[j], predictiveLookAhead, predictiveStrength);
                        steerPredict = Add(steerPredict, s);
                    }
                }

                Vector2 steerObs = { 0,0 };
                if (enableObstacleAvoid) steerObs = ObstacleAvoidance(a, obsCenters, obsRadii, obstacleLookAhead, obstacleStrength);

                Vector2 steerWall = { 0,0 };
                if (enableWallAvoid) steerWall = WallAvoidance(a, screenW, screenH, wallMargin, wallStrength);

                // Combine - either priority or weighted blend (Task3)
                Vector2 finalSteer = { 0,0 };
                if (usePriority) {
                    // priority order (highest -> lowest)
                    std::vector<Vector2> priorityForces;
                    priorityForces.push_back(Limit(Add(Scale(steerObs, 2.0f), Scale(steerWall, 1.8f)), a.maxForce)); // immediate danger
                    priorityForces.push_back(Limit(Add(Scale(steerPredict, 1.4f), Scale(steerSep, 1.2f)), a.maxForce)); // safety
                    priorityForces.push_back(Limit(Scale(steerPath, 0.9f), a.maxForce)); // navigation
                    finalSteer = PrioritySteering(priorityForces);
                }
                else {
                    // weighted blending
                    std::vector<std::pair<Vector2, float>> wforces;
                    wforces.push_back({ steerObs, 1.8f });
                    wforces.push_back({ steerWall, 1.4f });
                    wforces.push_back({ steerPredict, 1.2f });
                    wforces.push_back({ steerSep, 1.0f });
                    wforces.push_back({ steerPath, 0.9f });
                    finalSteer = WeightedBlend(wforces, a.maxForce);
                }

                // Apply as acceleration-like steering
                finalSteer = Limit(finalSteer, a.maxForce);
                a.vel = Add(a.vel, finalSteer);
                a.vel = Limit(a.vel, a.maxSpeed);
                a.pos = Add(a.pos, a.vel);

                // simple wrap-around prevention:
                if (a.pos.x < -60) a.pos.x = screenW + 60;
                if (a.pos.x > screenW + 60) a.pos.x = -60;
                if (a.pos.y < -60) a.pos.y = screenH + 60;
                if (a.pos.y > screenH + 60) a.pos.y = -60;
            }
        }

        // ---------- Drawing ----------
        BeginDrawing();
        ClearBackground(WHITE);

        // Draw path
        for (size_t i = 0;i < path.size();++i) {
            Vector2 a = path[i];
            Vector2 b = path[(i + 1) % path.size()];
            DrawLineEx(a, b, 2.0f, LIGHTGRAY);
            DrawCircleV(a, 6, DARKGRAY);
        }

        // Draw obstacles
        for (size_t i = 0;i < obsCenters.size();++i) {
            DrawCircleV(obsCenters[i], obsRadii[i], Fade(RED, 0.22f));
            DrawCircleLines((int)obsCenters[i].x, (int)obsCenters[i].y, obsRadii[i], RED);
            if (drawDebug) {
                DrawText(TextFormat("Obs %d", (int)i), (int)obsCenters[i].x - 18, (int)obsCenters[i].y - (int)obsRadii[i] - 18, 10, DARKGRAY);
            }
        }

        // Draw either single agent (Task1) or agents (Task2)
        if (singleAgentMode) {
            // draw target
            DrawCircleV(target, 7, DARKBLUE);

            // draw debug circle at player pos so you can see it even if triangle color blends
            if (drawDebug) DrawCircleLines((int)player.pos.x, (int)player.pos.y, 18, Fade(BLACK, 0.15f));

            // draw player triangle
            DrawAgentTriangle(player.pos, player.vel, ORANGE);

            if (drawDebug) {
                DrawText(TextFormat("Single-agent mode: %s %s",
                    (singleMode == 1 ? "Seek" : singleMode == 2 ? "Flee" : singleMode == 3 ? "Pursue" : singleMode == 4 ? "Evade" : singleMode == 5 ? "Arrive" : "Wander"),
                    singleCombine ? "(Combining ON - B)" : ""
                ), 20, 20, 36, BLACK);
                DrawText("Use 1..6 to change behavior. TAB to toggle mode. D debug toggle. P switch multi-agent combining. B toggle single-agent combine demo.", 30, 64, 24, DARKGRAY);
                // draw previous mouse velocity
                DrawLineEx(target, Add(target, Scale(mouseVel, 3.0f)), 2.0f, GRAY);
                
            }
        }
        else {
            // draw each agent
            for (const Agent& a : agents) {
                if (drawDebug) DrawCircleLines((int)a.pos.x, (int)a.pos.y, (int)separationRadius, Fade(DARKBLUE, 0.25f));
                DrawAgentTriangle(a.pos, a.vel, a.color);
                if (drawDebug) DrawLineEx(a.pos, Add(a.pos, Scale(a.vel, 18.0f)), 3.0f, DARKGRAY);
            }
            // UI text
            DrawText(TextFormat("Multi-agent mode (Task2). Agents: %d", (int)agents.size()), 30, 30, 48, BLACK);
            DrawText("Toggles: 1 Path  2 Separation  3 Predictive  4 ObsAvoid  5 WallAvoid  D Debug  P Priority/Weighted  TAB single/multi", 20, 64, 24, DARKGRAY);
            DrawText(TextFormat("Path:%s  Sep:%s  Predict:%s  Obs:%s  Wall:%s  Combining:%s",
                enablePathFollowing ? "ON" : "OFF",
                enableSeparation ? "ON" : "OFF",
                enablePredictiveAvoid ? "ON" : "OFF",
                enableObstacleAvoid ? "ON" : "OFF",
                enableWallAvoid ? "ON" : "OFF",

                usePriority ? "PRIORITY" : "WEIGHTED"
            ), 10, 54, 12, DARKGRAY);
        }

        // Legend/pause
        DrawText("Press ESC to exit.", screenW - 150, screenH - 28, 12, DARKGRAY);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}

