from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": True
})

import math
import time
import numpy as np
import omni.kit.app

from stable_baselines3 import PPO
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils.viewports import set_camera_view
from pxr import UsdLux


WAIT = 0
APPROACH = 1
RETREAT = 2
ATTACK = 3
BLOCK = 4
DODGE = 5

ACTION_NAMES = {
    WAIT: "WAIT",
    APPROACH: "APPROACH",
    RETREAT: "RETREAT",
    ATTACK: "ATTACK",
    BLOCK: "BLOCK",
    DODGE: "DODGE",
}


# --------------------------------------------------
# WebRTC livestream
# --------------------------------------------------

ext_manager = omni.kit.app.get_app().get_extension_manager()

for extension in [
    "omni.kit.livestream.core",
    "omni.kit.livestream.webrtc",
    "omni.kit.livestream.app",
]:
    if not ext_manager.is_extension_enabled(extension):
        ext_manager.set_extension_enabled_immediate(
            extension,
            True
        )

for _ in range(10):
    simulation_app.update()


def move_toward(current, target, amount):
    direction = target - current
    distance = np.linalg.norm(direction[:2])

    if distance < 1e-6:
        return current.copy()

    direction = direction / distance

    new_position = current.copy()
    new_position[0] += direction[0] * amount
    new_position[1] += direction[1] * amount

    return new_position


def move_away(current, target, amount):
    direction = current - target
    distance = np.linalg.norm(direction[:2])

    if distance < 1e-6:
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction = direction / distance

    new_position = current.copy()
    new_position[0] += direction[0] * amount
    new_position[1] += direction[1] * amount

    return new_position


def main():

    print("Loading PPO V2.2...")

    model = PPO.load(
        "/workspace/ai-fight-arena/training/alpha_policy_v22.zip",
        device="cpu"
    )

    print("PPO V2.2 loaded.")

    world = World(
        stage_units_in_meters=1.0
    )

    world.scene.add_default_ground_plane()

    world.scene.add(
        FixedCuboid(
            prim_path="/World/ArenaPlatform",
            name="arena_platform",
            position=np.array([0.0, 0.0, 0.05]),
            scale=np.array([8.0, 4.0, 0.1]),
            color=np.array([0.20, 0.20, 0.20]),
        )
    )

    alpha = world.scene.add(
        DynamicCuboid(
            prim_path="/World/Alpha",
            name="alpha",
            position=np.array([-2.0, 0.0, 0.65]),
            scale=np.array([0.6, 0.6, 1.2]),
            color=np.array([0.10, 0.35, 0.95]),
            mass=10.0,
        )
    )

    bravo = world.scene.add(
        DynamicCuboid(
            prim_path="/World/Bravo",
            name="bravo",
            position=np.array([2.0, 0.0, 0.65]),
            scale=np.array([0.6, 0.6, 1.2]),
            color=np.array([0.95, 0.15, 0.15]),
            mass=10.0,
        )
    )

    stage = world.stage

    light = UsdLux.DistantLight.Define(
        stage,
        "/World/Sun"
    )

    light.CreateIntensityAttr(3000.0)
    light.CreateAngleAttr(1.0)

    world.reset()

    set_camera_view(
        eye=np.array([0.0, -9.0, 5.5]),
        target=np.array([0.0, 0.0, 0.5]),
        camera_prim_path="/OmniverseKit_Persp"
    )

    # --------------------------------------------------
    # Fight state
    # --------------------------------------------------

    alpha_health = 100.0
    bravo_health = 100.0

    alpha_stamina = 100.0
    bravo_stamina = 100.0

    alpha_cooldown = 0
    bravo_cooldown = 0

    previous_alpha_action = WAIT
    previous_bravo_action = WAIT

    bravo_attack_telegraph = 0
    bravo_telegraph_pending = False

    bravo_aggression = 0.70

    # Persistent defensive state
    alpha_block_steps = 0
    alpha_dodge_steps = 0

    fight_over = False

    last_decision_time = time.monotonic()

    print("========================================")
    print("AI FIGHT ARENA - ISAAC SIM PPO V2.2")
    print("Alpha = BLUE PPO agent")
    print("Bravo = RED rule-based opponent")
    print("========================================")

    while simulation_app.is_running():

        world.step(render=True)

        if fight_over:
            continue

        now = time.monotonic()

        if now - last_decision_time < 1.0:
            continue

        last_decision_time = now

        alpha_position, _ = alpha.get_world_pose()
        bravo_position, _ = bravo.get_world_pose()

        alpha_position = np.array(
            alpha_position,
            dtype=np.float32
        )

        bravo_position = np.array(
            bravo_position,
            dtype=np.float32
        )

        dx = bravo_position[0] - alpha_position[0]
        dy = bravo_position[1] - alpha_position[1]

        distance = math.sqrt(
            dx * dx +
            dy * dy
        )

        # ----------------------------------------------
        # Persistent defensive timers
        # ----------------------------------------------

        alpha_blocking = alpha_block_steps > 0
        alpha_dodging = alpha_dodge_steps > 0

        alpha_block_steps = max(
            0,
            alpha_block_steps - 1
        )

        alpha_dodge_steps = max(
            0,
            alpha_dodge_steps - 1
        )

        # ----------------------------------------------
        # Cooldowns + stamina recovery
        # ----------------------------------------------

        alpha_cooldown = max(
            0,
            alpha_cooldown - 1
        )

        bravo_cooldown = max(
            0,
            bravo_cooldown - 1
        )

        alpha_stamina = min(
            100.0,
            alpha_stamina + 5.0
        )

        bravo_stamina = min(
            100.0,
            bravo_stamina + 5.0
        )

        # ----------------------------------------------
        # Bravo behavior
        # ----------------------------------------------

        bravo_attack_telegraph = 0

        if distance > 1.0:

            bravo_position = move_toward(
                bravo_position,
                alpha_position,
                0.30
            )

            bravo.set_world_pose(
                position=bravo_position
            )

            previous_bravo_action = APPROACH
            bravo_telegraph_pending = False

        else:

            if not bravo_telegraph_pending:

                bravo_attack_telegraph = 1
                bravo_telegraph_pending = True

                previous_bravo_action = WAIT

                print(
                    "BRAVO TELEGRAPHS ATTACK"
                )

            else:

                bravo_telegraph_pending = False
                previous_bravo_action = ATTACK

        # ----------------------------------------------
        # PPO observation
        # ----------------------------------------------

        observation = np.array(
            [
                alpha_health,
                bravo_health,
                distance,
                alpha_stamina,
                bravo_stamina,
                float(alpha_cooldown),
                float(bravo_cooldown),
                float(bravo_attack_telegraph),
                float(previous_alpha_action),
                float(previous_bravo_action),
                float(bravo_aggression),
            ],
            dtype=np.float32
        )

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        action = int(action)
        previous_alpha_action = action

        print(
            f"DIST={distance:.2f} | "
            f"Alpha HP={alpha_health:.1f} | "
            f"Bravo HP={bravo_health:.1f} | "
            f"STA={alpha_stamina:.0f} | "
            f"TEL={bravo_attack_telegraph} | "
            f"BLOCK={int(alpha_blocking)} | "
            f"DODGE={int(alpha_dodging)} | "
            f"PPO={ACTION_NAMES[action]}"
        )

        # ----------------------------------------------
        # Alpha action
        # ----------------------------------------------

        if action == WAIT:

            pass

        elif action == APPROACH:

            alpha_position = move_toward(
                alpha_position,
                bravo_position,
                0.35
            )

            alpha.set_world_pose(
                position=alpha_position
            )

        elif action == RETREAT:

            alpha_position = move_away(
                alpha_position,
                bravo_position,
                0.25
            )

            alpha.set_world_pose(
                position=alpha_position
            )

        elif action == BLOCK:

            if alpha_stamina >= 10:

                alpha_stamina -= 10

                # Active now + next decision
                alpha_block_steps = 2

                print(
                    "ALPHA BLOCKS - GUARD ACTIVE"
                )

        elif action == DODGE:

            if alpha_stamina >= 15:

                alpha_stamina -= 15

                # Active now + next decision
                alpha_dodge_steps = 2

                alpha_position = move_away(
                    alpha_position,
                    bravo_position,
                    0.45
                )

                alpha.set_world_pose(
                    position=alpha_position
                )

                print(
                    "ALPHA DODGES - EVASION ACTIVE"
                )

        elif action == ATTACK:

            if (
                distance <= 1.0 and
                alpha_cooldown == 0 and
                alpha_stamina >= 20
            ):

                alpha_stamina -= 20
                alpha_cooldown = 2

                bravo_health = max(
                    0.0,
                    bravo_health - 20.0
                )

                print(
                    f"ALPHA ATTACKS - "
                    f"Bravo HP={bravo_health:.1f}"
                )

            elif distance > 1.0:

                alpha_position = move_toward(
                    alpha_position,
                    bravo_position,
                    0.35
                )

                alpha.set_world_pose(
                    position=alpha_position
                )

        # ----------------------------------------------
        # Re-check defensive state AFTER Alpha acts
        # ----------------------------------------------

        alpha_blocking = alpha_block_steps > 0
        alpha_dodging = alpha_dodge_steps > 0

        # ----------------------------------------------
        # Resolve Bravo attack
        # ----------------------------------------------

        if (
            previous_bravo_action == ATTACK and
            distance <= 1.0 and
            bravo_cooldown == 0 and
            bravo_stamina >= 20
        ):

            bravo_stamina -= 20
            bravo_cooldown = 2

            if alpha_dodging:

                print(
                    "ALPHA DODGED BRAVO ATTACK"
                )

            else:

                damage = 15.0

                if alpha_blocking:

                    damage *= 0.45

                    print(
                        "ALPHA BLOCKED PART OF "
                        "BRAVO ATTACK"
                    )

                alpha_health = max(
                    0.0,
                    alpha_health - damage
                )

                print(
                    f"BRAVO ATTACKS - "
                    f"Alpha HP={alpha_health:.1f} "
                    f"(damage={damage:.1f})"
                )

        # ----------------------------------------------
        # KO handling
        # ----------------------------------------------

        if bravo_health <= 0:

            print("==============================")
            print("KO - PPO V2.2 ALPHA WINS")
            print("==============================")

            fight_over = True

        elif alpha_health <= 0:

            print("==============================")
            print("KO - BRAVO WINS")
            print("==============================")

            fight_over = True

    simulation_app.close()


if __name__ == "__main__":
    main()
