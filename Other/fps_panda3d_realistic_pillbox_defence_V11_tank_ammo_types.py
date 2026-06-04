# fps_panda3d_realistic_pillbox_defence_V10_armour_grenades_vehicle_health.py
# BUILD_MARKER: ARMOUR_ZONES_GRENADES_VEHICLE_HEALTH_V10_2026_05_29
#
# Install:
#   pip install panda3d
#
# Run:
#   python fps_panda3d_realistic_pillbox_defence_V7_rear_door_camera_movement_rpg.py
#
# Controls:
#   W = forward
#   S = backward
#   A = strafe left
#   D = strafe right
#   Mouse = look
#   Left click = fire
#   R = reload
#   Q = toggle scope
#   E = interact: leave/enter pillbox, enter/exit tank/APC
#   T = test explosion
#   G = throw grenade
#   1/2/3/4 = switch weapon
#   Space = jump / APC LMG fire when driving APC
#   Shift = sprint
#   Esc = pause menu
#
# Build marker: V11_TANK_HE_AP_AMMO_TYPES

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.gui.OnscreenText import OnscreenText

from panda3d.core import (
    Vec3, Vec4, Point3,
    WindowProperties,
    AmbientLight, DirectionalLight,
    Geom, GeomNode, GeomTriangles,
    GeomVertexFormat, GeomVertexData,
    GeomVertexWriter,
    LineSegs,
    TextNode,
)

import math
import random
import os
import sys
from dataclasses import dataclass


# ----------------------------
# Geometry helpers
# ----------------------------

def make_box(parent, name, size=(1, 1, 1), color=(1, 1, 1, 1), pos=(0, 0, 0), hpr=(0, 0, 0)):
    sx, sy, sz = size[0] / 2, size[1] / 2, size[2] / 2

    vertices = [
        (-sx, -sy, -sz), ( sx, -sy, -sz), ( sx,  sy, -sz), (-sx,  sy, -sz),
        (-sx, -sy,  sz), ( sx, -sy,  sz), ( sx,  sy,  sz), (-sx,  sy,  sz),
    ]

    faces = [
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    ]

    fmt = GeomVertexFormat.getV3c4()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)

    vertex = GeomVertexWriter(vdata, "vertex")
    colour = GeomVertexWriter(vdata, "color")

    for v in vertices:
        vertex.addData3(*v)
        colour.addData4(*color)

    tris = GeomTriangles(Geom.UHStatic)

    for a, b, c, d in faces:
        tris.addVertices(a, b, c)
        tris.addVertices(a, c, d)

    geom = Geom(vdata)
    geom.addPrimitive(tris)

    node = GeomNode(name)
    node.addGeom(geom)

    np = parent.attachNewNode(node)
    np.setPos(*pos)
    np.setHpr(*hpr)
    return np


def make_sphere(game, parent, name, pos, scale, color):
    sphere = game.loader.loadModel("models/misc/sphere")
    sphere.reparentTo(parent)
    sphere.setName(name)
    sphere.setPos(pos)
    if isinstance(scale, tuple):
        sphere.setScale(*scale)
    else:
        sphere.setScale(scale)
    sphere.setColor(*color)
    return sphere


def make_expanding_dome_mesh(parent, name="real-expanding-half-dome", color=(1.0, 0.45, 0.05, 0.70), rings=10, segments=48):
    """
    Procedural unit half-dome mesh with its base on z=0 and top at z=1.
    It is deliberately not a sphere model, so scaling it every frame is obvious.
    """
    fmt = GeomVertexFormat.getV3c4()
    vdata = GeomVertexData(name, fmt, Geom.UHDynamic)
    vertex = GeomVertexWriter(vdata, "vertex")
    colour = GeomVertexWriter(vdata, "color")

    for r in range(rings + 1):
        phi = (math.pi / 2.0) * r / rings
        ring_radius = math.sin(phi)
        z = math.cos(phi)
        for s in range(segments):
            a = math.tau * s / segments
            x = math.cos(a) * ring_radius
            y = math.sin(a) * ring_radius
            vertex.addData3(x, y, z)
            colour.addData4(*color)

    tris = GeomTriangles(Geom.UHDynamic)
    for r in range(rings):
        for s in range(segments):
            a = r * segments + s
            b = r * segments + ((s + 1) % segments)
            c = (r + 1) * segments + s
            d = (r + 1) * segments + ((s + 1) % segments)
            tris.addVertices(a, c, b)
            tris.addVertices(b, c, d)

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    np = parent.attachNewNode(node)
    np.setTwoSided(True)
    np.setTransparency(True)
    np.setLightOff()
    np.setDepthWrite(False)
    np.setColor(*color)
    return np



def make_projectile_visual(game, parent, weapon, pos, direction):
    """
    Creates a more realistic projectile visual.
    Normal rounds are small bright bullet cores with a faint tracer line.
    RPG rounds are rocket-shaped: dark body, red nose, fins, and exhaust glow.
    """
    root = parent.attachNewNode("projectile-root")
    root.setPos(pos)

    if weapon.name == "Grenade":
        body = make_sphere(game, root, "grenade-body", Point3(0, 0, 0), (0.18, 0.18, 0.18), (0.16, 0.20, 0.13, 1))
        make_box(root, "grenade-spoon", size=(0.045, 0.18, 0.025), color=(0.05, 0.05, 0.04, 1), pos=(0.05, 0, 0.16))
        make_box(root, "grenade-band", size=(0.25, 0.035, 0.035), color=(0.04, 0.05, 0.035, 1), pos=(0, 0.02, 0.00))
        return root

    if weapon.name == "Tank AP":
        # AP shell: long, fast dart / sabot style, less fiery than HE.
        make_box(root, "ap-dart-core", size=(0.055, 0.92, 0.055), color=(0.70, 0.72, 0.68, 1), pos=(0, 0, 0))
        make_box(root, "ap-dart-tip", size=(0.065, 0.20, 0.065), color=(0.88, 0.84, 0.70, 1), pos=(0, 0.56, 0))
        make_box(root, "ap-sabot-left", size=(0.10, 0.24, 0.035), color=(0.28, 0.30, 0.28, 1), pos=(-0.09, -0.18, 0))
        make_box(root, "ap-sabot-right", size=(0.10, 0.24, 0.035), color=(0.28, 0.30, 0.28, 1), pos=(0.09, -0.18, 0))
        tracer = LineSegs("ap-tracer")
        tracer.setThickness(2)
        tracer.setColor(0.95, 0.92, 0.72, 0.60)
        tracer.moveTo(0, -0.68, 0)
        tracer.drawTo(0, -0.16, 0)
        root.attachNewNode(tracer.create())

    elif weapon.name in ("RPG", "Tank HE", "Tank Cannon", "Enemy Tank Shell"):
        # Rocket/shell points along local +Y. We rotate the root with lookAt() each frame.
        make_box(root, "rocket-body", size=(0.14, 0.72, 0.14), color=(0.18, 0.18, 0.16, 1), pos=(0, 0, 0))
        make_box(root, "rocket-nose", size=(0.11, 0.20, 0.11), color=(0.65, 0.08, 0.05, 1), pos=(0, 0.46, 0))
        make_box(root, "rocket-fin-left", size=(0.04, 0.20, 0.22), color=(0.08, 0.08, 0.08, 1), pos=(-0.10, -0.32, 0))
        make_box(root, "rocket-fin-right", size=(0.04, 0.20, 0.22), color=(0.08, 0.08, 0.08, 1), pos=(0.10, -0.32, 0))
        make_box(root, "rocket-fin-top", size=(0.20, 0.20, 0.04), color=(0.08, 0.08, 0.08, 1), pos=(0, -0.32, 0.10))
        exhaust = make_sphere(game, root, "rocket-exhaust", Point3(0, -0.46, 0), 0.10, (1.0, 0.42, 0.05, 0.85))
        exhaust.setTransparency(True)
    else:
        # Small round bullet core plus a short tracer stroke behind it.
        make_sphere(game, root, "bullet-core", Point3(0, 0, 0), weapon.projectile_scale, weapon.projectile_color)
        trail = LineSegs("bullet-tracer")
        trail.setThickness(2)
        trail.setColor(1.0, 0.78, 0.22, 0.78)
        trail.moveTo(0, -0.36, 0)
        trail.drawTo(0, -0.04, 0)
        root.attachNewNode(trail.create())

    if direction.length() > 0:
        target = Point3(pos) + direction
        root.lookAt(target)

    return root


# ----------------------------
# Game data
# ----------------------------

@dataclass
class Weapon:
    name: str
    damage: float
    muzzle_velocity: float
    rpm: float
    mag_size: int
    reserve: int
    reload_time: float
    spread_deg: float
    gravity_scale: float
    pellets: int = 1
    splash_radius: float = 0.0
    splash_damage: float = 0.0
    projectile_scale: float = 0.05
    projectile_color: tuple = (1.0, 0.86, 0.25, 1.0)


@dataclass
class EnemyType:
    name: str
    health: float
    speed: float
    damage: float
    color: tuple
    size_scale: float
    reward: int


class Bullet:
    def __init__(self, pos, vel, weapon, node):
        self.pos = Point3(pos)
        self.vel = Vec3(vel)
        self.weapon = weapon
        self.node = node
        self.damage = weapon.damage
        self.gravity_scale = weapon.gravity_scale
        self.ttl = 4.0
        self.fuse = None
        self.bounces = 0
        self.source = None


class Enemy:
    def __init__(self, game, pos, enemy_type, level=1):
        self.game = game
        self.enemy_type = enemy_type
        self.root = game.render.attachNewNode("enemy")

        self.health = enemy_type.health + level * 9
        self.max_health = self.health
        self.speed = (enemy_type.speed + level * 0.04) * 0.90
        self.damage = enemy_type.damage + level * 0.35
        self.attack_cooldown = 0.0
        self.alive = True
        self.knockback_vel = Vec3(0, 0, 0)
        self.track_damage_timer = 0.0
        # Armoured tank mobs only return fire after the player shoots them.
        self.retaliation_timer = 0.0
        # V10: enemy tank return fire is intentionally slower/less dense.
        self.return_fire_cooldown = random.uniform(3.0, 5.0)

        s = enemy_type.size_scale
        self.root.setPos(pos)

        self.legs = make_box(
            self.root,
            "legs",
            size=(0.55 * s, 0.38 * s, 0.75 * s),
            color=(0.16, 0.20, 0.16, 1),
            pos=(0, 0, 0.38 * s)
        )

        self.chest = make_box(
            self.root,
            "chest",
            size=(0.72 * s, 0.46 * s, 0.78 * s),
            color=enemy_type.color,
            pos=(0, 0, 1.15 * s)
        )

        self.head = make_box(
            self.root,
            "head",
            size=(0.38 * s, 0.36 * s, 0.38 * s),
            color=(0.64, 0.46, 0.32, 1),
            pos=(0, 0, 1.78 * s)
        )

        # Larger, higher health bar so it is much easier to read at range.
        self.hp_bar_back = make_box(
            self.root,
            "hp-back",
            size=(1.32 * s, 0.075, 0.075),
            color=(0.02, 0.02, 0.02, 1),
            pos=(0, -0.36, 2.34 * s)
        )

        self.hp_bar_damage = make_box(
            self.root,
            "hp-damage-red",
            size=(1.22 * s, 0.082, 0.082),
            color=(0.70, 0.05, 0.04, 1),
            pos=(0, -0.405, 2.34 * s)
        )

        self.hp_bar = make_box(
            self.root,
            "hp-green",
            size=(1.22 * s, 0.095, 0.095),
            color=(0.05, 0.95, 0.12, 1),
            pos=(0, -0.46, 2.34 * s)
        )

        # V9: vehicle-type enemies should look like vehicles, not a soldier wearing a tank.
        # The simple humanoid hit-zone meshes still exist internally for non-vehicle mobs,
        # but are hidden for enemy APCs/tanks so no visible person appears.
        if self.enemy_type.name in ("APC", "Tank"):
            self.legs.hide()
            self.chest.hide()
            self.head.hide()

        if self.enemy_type.name == "APC":
            # Vehicle shell: the old hitbox still works, but the visual now reads as an armoured vehicle.
            make_box(self.root, "apc-chassis", size=(1.95 * s, 2.75 * s, 0.55 * s), color=(0.16, 0.18, 0.17, 1), pos=(0, 0, 0.52 * s))
            make_box(self.root, "apc-sloped-front", size=(1.60 * s, 0.42 * s, 0.38 * s), color=(0.22, 0.24, 0.22, 1), pos=(0, 1.36 * s, 0.78 * s), hpr=(0, 12, 0))
            make_box(self.root, "apc-turret", size=(0.88 * s, 0.72 * s, 0.38 * s), color=(0.13, 0.15, 0.14, 1), pos=(0, 0.18 * s, 1.02 * s))
            make_box(self.root, "apc-barrel", size=(0.13 * s, 1.25 * s, 0.13 * s), color=(0.08, 0.09, 0.08, 1), pos=(0, 0.95 * s, 1.05 * s))
            for wx in [-0.95, 0.95]:
                for wy in [-0.95, -0.32, 0.32, 0.95]:
                    make_sphere(self.game, self.root, "apc-wheel", Point3(wx * s, wy * s, 0.20 * s), (0.20 * s, 0.09 * s, 0.20 * s), (0.035, 0.035, 0.035, 1))

        if self.enemy_type.name == "Tank":
            # Armoured enemy tank. It is resistant to small-arms knockback and returns fire only after being hit.
            make_box(self.root, "enemy-tank-hull", size=(2.35 * s, 3.55 * s, 0.78 * s), color=(0.12, 0.15, 0.12, 1), pos=(0, 0, 0.48 * s))
            make_box(self.root, "enemy-tank-front-slope", size=(2.05 * s, 0.58 * s, 0.45 * s), color=(0.18, 0.21, 0.17, 1), pos=(0, 1.78 * s, 0.74 * s), hpr=(0, 10, 0))
            make_box(self.root, "enemy-tank-turret", size=(1.25 * s, 1.05 * s, 0.46 * s), color=(0.09, 0.12, 0.09, 1), pos=(0, 0.20 * s, 1.05 * s))
            make_box(self.root, "enemy-tank-gun", size=(0.16 * s, 2.15 * s, 0.16 * s), color=(0.045, 0.055, 0.045, 1), pos=(0, 1.55 * s, 1.08 * s))
            for tx in [-1.18, 1.18]:
                make_box(self.root, "enemy-tank-track", size=(0.34 * s, 3.65 * s, 0.38 * s), color=(0.025, 0.028, 0.025, 1), pos=(tx * s, 0, 0.20 * s))
                for wy in [-1.25, -0.62, 0.0, 0.62, 1.25]:
                    make_sphere(self.game, self.root, "enemy-tank-wheel", Point3(tx * s, wy * s, 0.20 * s), (0.17 * s, 0.06 * s, 0.17 * s), (0.015, 0.015, 0.015, 1))
            # Extra armour features so the enemy tank reads clearly as an unmanned vehicle.
            make_box(self.root, "enemy-tank-side-skirt-L", size=(0.16 * s, 3.50 * s, 0.34 * s), color=(0.075, 0.095, 0.075, 1), pos=(-1.43 * s, 0, 0.42 * s))
            make_box(self.root, "enemy-tank-side-skirt-R", size=(0.16 * s, 3.50 * s, 0.34 * s), color=(0.075, 0.095, 0.075, 1), pos=(1.43 * s, 0, 0.42 * s))
            make_box(self.root, "enemy-tank-engine-deck", size=(1.55 * s, 0.82 * s, 0.10 * s), color=(0.055, 0.070, 0.055, 1), pos=(0, -1.18 * s, 0.93 * s))
            make_box(self.root, "enemy-tank-hatch", size=(0.52 * s, 0.42 * s, 0.12 * s), color=(0.045, 0.055, 0.045, 1), pos=(0, -0.16 * s, 1.34 * s))
            make_box(self.root, "enemy-tank-muzzle-brake", size=(0.28 * s, 0.18 * s, 0.22 * s), color=(0.020, 0.025, 0.020, 1), pos=(0, 2.68 * s, 1.08 * s))
            # V10 higher-detail tank silhouette: ERA blocks, optics, exhaust, antennas.
            for px in [-0.66, -0.22, 0.22, 0.66]:
                make_box(self.root, "enemy-tank-era-front", size=(0.34 * s, 0.10 * s, 0.24 * s), color=(0.060, 0.080, 0.060, 1), pos=(px * s, 2.10 * s, 0.84 * s), hpr=(0, 10, 0))
            for side_x in [-1.30, 1.30]:
                for yy in [-1.20, -0.55, 0.10, 0.75, 1.40]:
                    make_box(self.root, "enemy-tank-side-era", size=(0.10 * s, 0.38 * s, 0.24 * s), color=(0.050, 0.070, 0.050, 1), pos=(side_x * s, yy * s, 0.73 * s))
            make_box(self.root, "enemy-tank-thermal-sight", size=(0.18 * s, 0.18 * s, 0.13 * s), color=(0.015, 0.030, 0.020, 1), pos=(0.48 * s, 0.40 * s, 1.36 * s))
            make_box(self.root, "enemy-tank-exhaust-left", size=(0.18 * s, 0.38 * s, 0.18 * s), color=(0.030, 0.030, 0.030, 1), pos=(-0.75 * s, -1.55 * s, 0.82 * s))
            make_box(self.root, "enemy-tank-exhaust-right", size=(0.18 * s, 0.38 * s, 0.18 * s), color=(0.030, 0.030, 0.030, 1), pos=(0.75 * s, -1.55 * s, 0.82 * s))
            make_box(self.root, "enemy-tank-antenna-base", size=(0.08 * s, 0.08 * s, 0.16 * s), color=(0.015, 0.015, 0.012, 1), pos=(-0.48 * s, -0.48 * s, 1.46 * s))
            make_box(self.root, "enemy-tank-antenna", size=(0.025 * s, 0.025 * s, 0.85 * s), color=(0.010, 0.010, 0.008, 1), pos=(-0.48 * s, -0.48 * s, 1.92 * s))

    def world_pos(self):
        return self.root.getPos(self.game.render)

    def separation_force(self):
        """Pushes mobs away from each other so they do not form one unreadable clump."""
        my_pos = self.world_pos()
        force = Vec3(0, 0, 0)

        desired_spacing = 2.2
        strength = 4.5

        for other in self.game.enemies:
            if other is self or not other.alive:
                continue

            other_pos = other.world_pos()
            offset = Vec3(my_pos.x - other_pos.x, my_pos.y - other_pos.y, 0)
            dist = offset.length()

            if 0.001 < dist < desired_spacing:
                offset.normalize()
                push = (desired_spacing - dist) / desired_spacing
                force += offset * push * strength

        return force

    def update(self, dt):
        if not self.alive:
            return

        pos = self.world_pos()

        if self.game.pillbox_health > 0:
            target = self.game.pillbox_pos
            attack_distance = self.game.pillbox_radius + 0.9
        else:
            target = self.game.player_pos
            attack_distance = 1.55

        to_target = Vec3(target.x - pos.x, target.y - pos.y, 0)
        dist = max(to_target.length(), 0.001)
        direction = to_target / dist

        heading = math.degrees(math.atan2(direction.x, direction.y))
        self.root.setH(heading)

        ai_velocity = Vec3(0, 0, 0)
        self.track_damage_timer = max(0.0, self.track_damage_timer - dt)

        if dist > attack_distance:
            side = Vec3(direction.y, -direction.x, 0)
            zigzag = math.sin(self.game.time * 2.0 + pos.x * 0.25) * 0.25
            separation = self.separation_force()
            speed_factor = 0.45 if self.track_damage_timer > 0.0 else 1.0
            ai_velocity = direction * self.speed * speed_factor + side * zigzag + separation

            if ai_velocity.length() > self.speed * 1.8:
                ai_velocity.normalize()
                ai_velocity *= self.speed * 1.8

        # Strong knockback should temporarily overpower the chase AI, otherwise
        # the mob appears to slide forward while being hit.
        horizontal_knock = Vec3(self.knockback_vel.x, self.knockback_vel.y, 0).length()
        if horizontal_knock > 0.75:
            ai_velocity *= 0.18

        # Knockback is a real velocity applied over time. It includes vertical
        # velocity, and gravity pulls mobs back down to the terrain.
        if self.knockback_vel.lengthSquared() > 0.0001 or pos.z > self.game.terrain_height(pos.x, pos.y) + 0.02:
            self.knockback_vel.z -= 9.81 * dt
        else:
            self.knockback_vel = Vec3(0, 0, 0)

        total_velocity = ai_velocity + self.knockback_vel
        new_x = pos.x + total_velocity.x * dt
        new_y = pos.y + total_velocity.y * dt
        new_z = pos.z + total_velocity.z * dt

        ground_z = self.game.terrain_height(new_x, new_y)
        if new_z <= ground_z:
            new_z = ground_z
            if self.knockback_vel.z < 0:
                self.knockback_vel.z = 0

        self.root.setPos(new_x, new_y, new_z)

        # Ground friction / air drag on knockback after integration.
        horizontal_decay = max(0.0, 1.0 - 2.6 * dt)
        self.knockback_vel.x *= horizontal_decay
        self.knockback_vel.y *= horizontal_decay
        self.knockback_vel.z *= max(0.0, 1.0 - 0.25 * dt)

        if abs(self.knockback_vel.x) < 0.02:
            self.knockback_vel.x = 0
        if abs(self.knockback_vel.y) < 0.02:
            self.knockback_vel.y = 0
        if abs(self.knockback_vel.z) < 0.02 and new_z <= ground_z + 0.02:
            self.knockback_vel.z = 0

        self.attack_cooldown = max(0.0, self.attack_cooldown - dt)
        self.return_fire_cooldown = max(0.0, self.return_fire_cooldown - dt)
        self.retaliation_timer = max(0.0, self.retaliation_timer - dt)

        if self.enemy_type.name == "Tank" and self.retaliation_timer > 0.0 and self.return_fire_cooldown <= 0.0:
            self.game.fire_enemy_tank_round(self)
            self.return_fire_cooldown = random.uniform(3.0, 5.0)

        # Recalculate distance after movement for attacking.
        pos2 = self.world_pos()
        attack_dist_now = math.sqrt((target.x - pos2.x) ** 2 + (target.y - pos2.y) ** 2)

        if attack_dist_now <= attack_distance and self.attack_cooldown <= 0.0:
            if self.game.pillbox_health > 0:
                self.game.damage_pillbox(self.damage)
            else:
                self.game.damage_player(self.damage)

            self.attack_cooldown = 0.85

        ratio = max(0.0, self.health / self.max_health)
        self.hp_bar.setScale(ratio, 1, 1)
        self.hp_bar.setX(-(1.0 - ratio) * 0.61)

        # Make the health bar face roughly toward the player horizontally.
        player = self.game.player_pos
        bar_pos = self.world_pos()
        bar_heading = math.degrees(math.atan2(player.x - bar_pos.x, player.y - bar_pos.y))
        self.hp_bar_back.setH(bar_heading)
        self.hp_bar_damage.setH(bar_heading)
        self.hp_bar.setH(bar_heading)

    def apply_damage(self, amount):
        self.health -= amount

        if self.health <= 0:
            self.die()

    def die(self):
        self.alive = False
        self.root.removeNode()
        self.game.kills += 1
        self.game.money += self.enemy_type.reward
        self.game.wave_kills += 1
        self.game.kill_streak += 1

        # Streak reward: small automatic repair and ammo trickle so defending
        # well feels useful, not just a scoreboard number.
        if self.game.kill_streak % 5 == 0:
            self.game.pillbox_health = min(self.game.pillbox_max_health, self.game.pillbox_health + 22)
            self.game.reserves[self.game.current_weapon] += max(3, self.game.weapons[self.game.current_weapon].mag_size // 3)
            self.game.set_status("5-KILL STREAK: +REPAIR +AMMO", 1.8)


# ----------------------------
# Main game
# ----------------------------

class FPSGame(ShowBase):
    def __init__(self):
        super().__init__()

        self.disableMouse()

        props = WindowProperties()
        props.setCursorHidden(True)
        self.win.requestProperties(props)

        self.center_x = self.win.getXSize() // 2
        self.center_y = self.win.getYSize() // 2
        self.win.movePointer(0, self.center_x, self.center_y)

        self.set_background_color(0.48, 0.68, 0.92, 1)

        self.time = 0.0
        self.wave = 1
        self.wave_kills = 0
        self.wave_goal = 12
        self.kills = 0
        self.money = 0
        self.game_over = False

        self.pillbox_radius = 4.2
        self.pillbox_pos = Point3(0, -24, self.terrain_height(0, -24))

        self.player_pos = Point3(
            self.pillbox_pos.x,
            self.pillbox_pos.y,
            self.pillbox_pos.z + 1.65
        )

        self.player_vel = Vec3(0, 0, 0)
        self.eye_height = 1.65
        self.grounded = False

        self.health = 100
        self.max_health = 100

        self.pillbox_health = 300
        self.pillbox_max_health = 300

        self.yaw = 0.0
        self.pitch = 0.0
        self.mouse_sensitivity = 0.12

        self.scoped = False
        self.normal_fov = 75
        self.scope_fov = 28

        self.keys = {
            "w": False,
            "a": False,
            "s": False,
            "d": False,
            "space": False,
            "shift": False,
            "arrow_left": False,
            "arrow_right": False,
            "arrow_up": False,
            "arrow_down": False,
        }

        self.mouse_down = False

        self.fire_timer = 0.0
        self.reload_timer = 0.0
        self.reload_total = 0.0
        self.reloading = False

        self.enemy_types = [
            EnemyType(
                name="Runner",
                health=55,
                speed=3.25,
                damage=4.2,
                color=(0.45, 0.12, 0.10, 1),
                size_scale=0.95,
                reward=15
            ),
            EnemyType(
                name="Bruiser",
                health=135,
                speed=1.55,
                damage=9.5,
                color=(0.20, 0.16, 0.12, 1),
                size_scale=1.25,
                reward=35
            ),
            EnemyType(
                name="Raider",
                health=80,
                speed=2.3,
                damage=6.5,
                color=(0.14, 0.22, 0.36, 1),
                size_scale=1.0,
                reward=22
            ),
            EnemyType(
                name="APC",
                health=320,
                speed=0.95,
                damage=18.0,
                color=(0.18, 0.20, 0.19, 1),
                size_scale=1.55,
                reward=95
            ),
            EnemyType(
                name="Tank",
                health=620,
                speed=0.62,
                damage=26.0,
                color=(0.11, 0.14, 0.11, 1),
                size_scale=1.82,
                reward=175
            ),
        ]

        self.weapons = [
            Weapon(
                name="Rifle",
                damage=34,
                muzzle_velocity=135,
                rpm=420,
                mag_size=20,
                reserve=120,
                reload_time=1.65,
                spread_deg=0.35,
                gravity_scale=0.28,
                projectile_scale=0.045,
                projectile_color=(1.0, 0.87, 0.25, 1.0)
            ),
            Weapon(
                name="SMG",
                damage=17,
                muzzle_velocity=85,
                rpm=760,
                mag_size=36,
                reserve=216,
                reload_time=1.35,
                spread_deg=1.15,
                gravity_scale=0.35,
                projectile_scale=0.04,
                projectile_color=(1.0, 0.78, 0.20, 1.0)
            ),
            Weapon(
                name="Shotgun",
                damage=13,
                muzzle_velocity=65,
                rpm=90,
                mag_size=7,
                reserve=42,
                reload_time=2.1,
                spread_deg=4.2,
                gravity_scale=0.45,
                pellets=9,
                projectile_scale=0.035,
                projectile_color=(1.0, 0.65, 0.18, 1.0)
            ),
            Weapon(
                name="RPG",
                # V10: RPG is a proper light anti-armour weapon: usually 2 direct hits kill a tank.
                damage=370,
                muzzle_velocity=58,
                rpm=28,
                mag_size=1,
                reserve=10,
                reload_time=2.7,
                spread_deg=0.10,
                gravity_scale=0.0,
                pellets=1,
                splash_radius=8.5,
                splash_damage=230,
                projectile_scale=0.18,
                projectile_color=(0.95, 0.18, 0.08, 1.0)
            ),
        ]

        self.current_weapon = 0
        self.magazines = [w.mag_size for w in self.weapons]
        self.reserves = [w.reserve for w in self.weapons]

        self.bullets = []
        self.enemies = []

        self.spawn_timer = 0.0
        self.hit_marker_timer = 0.0
        self.damage_flash_timer = 0.0

        # Extra gameplay systems.
        self.kill_streak = 0
        self.status_message = ""
        self.status_timer = 0.0
        self.mortar_cooldown = 0.0
        self.repair_cost = 75
        self.mortar_cost = 120
        self.mortar_targeting = False
        self.mortar_target_rel = Vec3(0, 36, 0)
        self.mortar_target_range = 52.0
        self.paused = False
        self.mouse_locked = True
        self.pause_menu_np = None
        self.wave_break = False
        self.wave_break_timer = 0.0
        self.wave_break_duration = 6.0
        self.next_wave_number = 1

        # V6: usable player vehicle and pillbox entry/exit state.
        self.in_pillbox = True
        self.in_vehicle = False
        self.vehicle_np = None
        self.vehicle_body_np = None
        self.vehicle_turret_np = None
        self.vehicle_barrel_np = None
        self.vehicle_pos = Point3(self.pillbox_pos.x + 8.5, self.pillbox_pos.y - 8.0, 0)
        self.vehicle_h = 18.0
        self.vehicle_speed = 0.0
        self.vehicle_fire_timer = 0.0
        self.vehicle_max_health = 520
        self.vehicle_health = self.vehicle_max_health
        # V11: selectable tank ammunition.
        # HE is the existing explosive shell style: slower, bigger blast, stronger knockback.
        # AP is a high-velocity armour-piercing round: much better against tanks, less splash/knockback.
        self.tank_ammo_index = 0
        self.tank_ammo_types = [
            Weapon(
                name="Tank HE",
                damage=340,
                muzzle_velocity=98,
                rpm=28,
                mag_size=1,
                reserve=999,
                reload_time=2.2,
                spread_deg=0.11,
                gravity_scale=0.09,
                pellets=1,
                splash_radius=8.0,
                splash_damage=230,
                projectile_scale=0.125,
                projectile_color=(1.0, 0.52, 0.12, 1.0),
            ),
            Weapon(
                name="Tank AP",
                damage=610,
                muzzle_velocity=168,
                rpm=24,
                mag_size=1,
                reserve=999,
                reload_time=2.7,
                spread_deg=0.055,
                gravity_scale=0.035,
                pellets=1,
                splash_radius=1.1,
                splash_damage=24,
                projectile_scale=0.075,
                projectile_color=(0.92, 0.88, 0.62, 1.0),
            ),
        ]
        self.tank_weapon = self.tank_ammo_types[self.tank_ammo_index]

        # V8: second driveable vehicle, an APC with a light machine gun.
        self.current_vehicle_kind = None
        self.apc_np = None
        self.apc_body_np = None
        self.apc_turret_np = None
        self.apc_barrel_np = None
        self.apc_pos = Point3(self.pillbox_pos.x - 8.5, self.pillbox_pos.y - 8.0, 0)
        self.apc_h = -18.0
        self.apc_speed = 0.0
        self.apc_lmg_fire_timer = 0.0
        self.apc_lmg_mag = 200
        self.apc_lmg_reserve = 600
        self.apc_lmg_reloading = False
        self.apc_lmg_reload_timer = 0.0
        self.apc_max_health = 360
        self.apc_health = self.apc_max_health
        self.grenade_count = 6
        self.grenade_cooldown = 0.0
        self.grenade_weapon = Weapon(
            name="Grenade",
            damage=12,
            muzzle_velocity=24,
            rpm=30,
            mag_size=1,
            reserve=6,
            reload_time=0.0,
            spread_deg=1.2,
            gravity_scale=1.0,
            pellets=1,
            splash_radius=6.2,
            splash_damage=145,
            projectile_scale=0.18,
            projectile_color=(0.20, 0.26, 0.18, 1.0),
        )
        self.apc_lmg_weapon = Weapon(
            name="APC LMG",
            damage=15,
            muzzle_velocity=145,
            rpm=720,  # 12 rounds/second
            mag_size=200,
            reserve=600,
            reload_time=5.0,
            spread_deg=0.85,
            gravity_scale=0.16,
            pellets=1,
            splash_radius=0.0,
            splash_damage=0,
            projectile_scale=0.033,
            projectile_color=(1.0, 0.88, 0.30, 1.0),
        )

        self.enemy_tank_shell_weapon = Weapon(
            name="Enemy Tank Shell",
            damage=88,
            muzzle_velocity=70,
            rpm=18,
            mag_size=1,
            reserve=999,
            reload_time=3.2,
            spread_deg=0.2,
            gravity_scale=0.06,
            pellets=1,
            splash_radius=5.5,
            splash_damage=95,
            projectile_scale=0.11,
            projectile_color=(1.0, 0.35, 0.10, 1.0),
        )

        self.reload_ring_np = None
        self.minimap_np = None
        self.scope_np = None
        self.active_explosions = []

        self.setup_lighting()
        self.create_terrain()
        self.create_round_pillbox()
        self.create_cover_and_details()
        self.create_player_vehicle()
        self.create_player_apc()
        self.create_hud()
        self.bind_keys()

        self.cam.node().getLens().setFov(self.normal_fov)
        self.camera.setPos(self.player_pos)
        self.camera.setHpr(self.yaw, self.pitch, 0)

        self.taskMgr.add(self.update, "update")

    # ----------------------------
    # Terrain
    # ----------------------------

    def terrain_height(self, x, y):
        # V5: flatter, wider battlefield. Still has shallow trenches and gentle cover.
        hills = (
            0.38 * math.sin(x * 0.050) * math.cos(y * 0.040)
            + 0.14 * math.sin((x + y) * 0.075)
            + 0.07 * math.sin(x * 0.130)
        )

        trench_front = -0.42 * math.exp(-((y + 2) ** 2) / 10.5)
        trench_side = -0.30 * math.exp(-((x - 15) ** 2) / 10.5)

        return hills + trench_front + trench_side

    def create_terrain(self):
        size = 110
        # Higher-poly terrain mesh.
        step = 1
        half = size // 2

        fmt = GeomVertexFormat.getV3n3c4()
        vdata = GeomVertexData("terrain", fmt, Geom.UHStatic)

        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        color = GeomVertexWriter(vdata, "color")

        points = []

        for iy in range(-half, half + 1, step):
            row = []
            for ix in range(-half, half + 1, step):
                z = self.terrain_height(ix, iy)
                row.append((ix, iy, z))
            points.append(row)

        rows = len(points)
        cols = len(points[0])

        for r in range(rows):
            for c in range(cols):
                x, y, z = points[r][c]

                hx1 = self.terrain_height(x + 1, y)
                hx0 = self.terrain_height(x - 1, y)
                hy1 = self.terrain_height(x, y + 1)
                hy0 = self.terrain_height(x, y - 1)

                n = Vec3(hx0 - hx1, hy0 - hy1, 3.0)
                n.normalize()

                vertex.addData3(x, y, z)
                normal.addData3(n)

                if z < -0.8:
                    colour = Vec4(0.30, 0.22, 0.14, 1)
                elif z > 1.2:
                    colour = Vec4(0.27, 0.35, 0.19, 1)
                else:
                    colour = Vec4(0.22, 0.42, 0.20, 1)

                color.addData4(colour)

        tris = GeomTriangles(Geom.UHStatic)

        for r in range(rows - 1):
            for c in range(cols - 1):
                i = r * cols + c
                tris.addVertices(i, i + 1, i + cols)
                tris.addVertices(i + 1, i + cols + 1, i + cols)

        geom = Geom(vdata)
        geom.addPrimitive(tris)

        node = GeomNode("terrain")
        node.addGeom(geom)

        self.terrain = self.render.attachNewNode(node)
        self.terrain.setTwoSided(True)

    # ----------------------------
    # Pillbox/world
    # ----------------------------

    def create_round_pillbox(self):
        x = self.pillbox_pos.x
        y = self.pillbox_pos.y
        z = self.pillbox_pos.z

        stone = (0.34, 0.34, 0.34, 1)
        stone_dark = (0.24, 0.24, 0.24, 1)

        make_sphere(
            self,
            self.render,
            "pillbox-floor",
            Point3(x, y, z + 0.12),
            (self.pillbox_radius, self.pillbox_radius, 0.10),
            (0.25, 0.25, 0.24, 1)
        )

        segments = 24
        wall_height = 1.10

        for i in range(segments):
            angle = math.tau * i / segments

            # V7: keep normal vision gaps, but make a much wider open rear doorway.
            # Rear is negative Y, angle pi in this coordinate system.
            rear_gap = abs(math.atan2(math.sin(angle - math.pi), math.cos(angle - math.pi))) < math.radians(34)

            if i % 4 == 0 or rear_gap:
                continue

            px = x + math.sin(angle) * self.pillbox_radius
            py = y + math.cos(angle) * self.pillbox_radius
            heading = -math.degrees(angle)

            make_box(
                self.render,
                "round-wall-segment",
                size=(1.15, 0.42, wall_height),
                color=stone,
                pos=(px, py, z + wall_height / 2),
                hpr=(heading, 0, 0)
            )

        for i in range(12):
            angle = math.tau * i / 12

            # Do not block the rear doorway with inner stones.
            rear_gap = abs(math.atan2(math.sin(angle - math.pi), math.cos(angle - math.pi))) < math.radians(38)
            if rear_gap:
                continue

            px = x + math.sin(angle) * (self.pillbox_radius - 1.1)
            py = y + math.cos(angle) * (self.pillbox_radius - 1.1)
            heading = -math.degrees(angle)

            make_box(
                self.render,
                "inner-stone",
                size=(0.75, 0.32, 0.30),
                color=stone_dark,
                pos=(px, py, z + 0.28),
                hpr=(heading, 0, 0)
            )

        # V7: visible rear exit threshold/gap. Walk through this opening to leave.
        make_box(
            self.render,
            "rear-door-threshold",
            size=(2.9, 0.55, 0.13),
            color=(0.10, 0.42, 0.12, 1),
            pos=(x, y - self.pillbox_radius - 0.08, z + 0.075),
            hpr=(0, 0, 0)
        )

    def create_cover_and_details(self):
        random.seed(4)

        self.cover_positions = [
            (-16, 0), (-10, 2), (-4, 1), (5, 0), (12, 2), (18, -1),
            (-22, 14), (-14, 17), (-6, 15), (4, 16), (13, 18), (22, 15),
            (-18, 30), (-7, 33), (8, 31), (18, 34),
            (-28, -2), (28, -2), (-30, 12), (30, 12)
        ]

        for x, y in self.cover_positions:
            z = self.terrain_height(x, y) + 0.35
            block = make_box(
                self.render,
                "cover",
                size=(2.4, 0.75, 0.7),
                color=(0.42, 0.34, 0.22, 1),
                pos=(x, y, z)
            )
            block.setH(random.choice([-25, -10, 0, 15, 30, 90]))

        for x, y in [(-35, 25), (35, 25), (-35, 45), (35, 45)]:
            z = self.terrain_height(x, y)
            make_box(
                self.render,
                "marker-pole",
                size=(0.22, 0.22, 4.0),
                color=(0.12, 0.12, 0.12, 1),
                pos=(x, y, z + 2.0)
            )

    def create_player_vehicle(self):
        """Create a driveable light tank/APC parked outside the pillbox."""
        self.vehicle_pos.z = self.terrain_height(self.vehicle_pos.x, self.vehicle_pos.y) + 0.55
        root = self.render.attachNewNode("player-driveable-tank")
        self.vehicle_np = root

        # Small prepared parking pad / ramp so the exit route reads clearly.
        pad_z = self.terrain_height(self.vehicle_pos.x, self.vehicle_pos.y) + 0.04
        make_box(self.render, "vehicle-parking-pad", size=(7.0, 5.5, 0.10), color=(0.18, 0.18, 0.17, 1), pos=(self.vehicle_pos.x, self.vehicle_pos.y, pad_z))
        make_box(self.render, "pillbox-exit-ramp", size=(2.4, 4.4, 0.12), color=(0.25, 0.24, 0.21, 1), pos=(self.pillbox_pos.x, self.pillbox_pos.y - self.pillbox_radius - 1.7, self.terrain_height(self.pillbox_pos.x, self.pillbox_pos.y - self.pillbox_radius - 1.7) + 0.06))

        self.vehicle_body_np = make_box(root, "tank-hull", size=(2.8, 4.2, 0.85), color=(0.14, 0.17, 0.14, 1), pos=(0, 0, 0.45))
        make_box(root, "tank-front-slope", size=(2.45, 0.70, 0.55), color=(0.20, 0.23, 0.19, 1), pos=(0, 2.10, 0.75), hpr=(0, 10, 0))
        make_box(root, "tank-rear-plate", size=(2.45, 0.45, 0.65), color=(0.11, 0.13, 0.11, 1), pos=(0, -2.05, 0.62))

        self.vehicle_turret_np = root.attachNewNode("tank-turret-root")
        self.vehicle_turret_np.setPos(0, 0.25, 1.05)
        make_box(self.vehicle_turret_np, "tank-turret", size=(1.45, 1.15, 0.55), color=(0.10, 0.13, 0.10, 1), pos=(0, 0, 0))
        self.vehicle_barrel_np = make_box(self.vehicle_turret_np, "tank-barrel", size=(0.18, 2.60, 0.18), color=(0.055, 0.065, 0.055, 1), pos=(0, 1.65, 0.04))
        make_box(self.vehicle_turret_np, "barrel-muzzle", size=(0.24, 0.28, 0.24), color=(0.025, 0.025, 0.025, 1), pos=(0, 3.02, 0.04))

        # Tracks and wheels.
        for sx in [-1, 1]:
            make_box(root, "tank-track", size=(0.46, 4.35, 0.44), color=(0.035, 0.038, 0.035, 1), pos=(sx * 1.46, 0, 0.20))
            for wy in [-1.55, -0.80, 0.0, 0.80, 1.55]:
                make_sphere(self, root, "tank-roadwheel", Point3(sx * 1.47, wy, 0.20), (0.22, 0.08, 0.22), (0.02, 0.02, 0.02, 1))

        # A small blue marker on top helps you find your own vehicle.
        make_box(root, "friendly-marker", size=(0.42, 0.42, 0.10), color=(0.05, 0.35, 1.0, 1), pos=(0, -0.55, 1.48))
        # V10 detail pass: ERA blocks, optics, machine-gun nub, exhausts, towing lugs.
        for px in [-0.85, -0.42, 0.0, 0.42, 0.85]:
            make_box(root, "friendly-tank-front-era", size=(0.34, 0.12, 0.24), color=(0.065, 0.090, 0.065, 1), pos=(px, 2.48, 0.88), hpr=(0, 10, 0))
        for sx in [-1, 1]:
            for yy in [-1.35, -0.65, 0.05, 0.75, 1.45]:
                make_box(root, "friendly-tank-side-era", size=(0.12, 0.42, 0.24), color=(0.060, 0.085, 0.060, 1), pos=(sx * 1.72, yy, 0.76))
            make_box(root, "friendly-tank-exhaust", size=(0.18, 0.40, 0.18), color=(0.035, 0.035, 0.033, 1), pos=(sx * 0.78, -2.18, 0.88))
        make_box(self.vehicle_turret_np, "tank-optic-box", size=(0.18, 0.18, 0.14), color=(0.02, 0.04, 0.025, 1), pos=(0.52, 0.25, 0.32))
        make_box(self.vehicle_turret_np, "coaxial-mg", size=(0.045, 1.25, 0.045), color=(0.012, 0.014, 0.012, 1), pos=(-0.30, 1.08, 0.09))
        self.update_vehicle_visual()

    def create_player_apc(self):
        """Create a separate driveable APC with a 200-round light machine gun."""
        self.apc_pos.z = self.terrain_height(self.apc_pos.x, self.apc_pos.y) + 0.48
        root = self.render.attachNewNode("player-driveable-apc")
        self.apc_np = root

        pad_z = self.terrain_height(self.apc_pos.x, self.apc_pos.y) + 0.04
        make_box(self.render, "apc-parking-pad", size=(6.2, 5.2, 0.10), color=(0.16, 0.17, 0.16, 1), pos=(self.apc_pos.x, self.apc_pos.y, pad_z))

        self.apc_body_np = make_box(root, "apc-hull", size=(2.6, 4.0, 0.78), color=(0.13, 0.16, 0.15, 1), pos=(0, 0, 0.42))
        make_box(root, "apc-front-slope", size=(2.25, 0.62, 0.48), color=(0.20, 0.23, 0.21, 1), pos=(0, 2.02, 0.70), hpr=(0, 10, 0))
        make_box(root, "apc-rear", size=(2.25, 0.42, 0.55), color=(0.10, 0.12, 0.11, 1), pos=(0, -1.98, 0.60))

        self.apc_turret_np = root.attachNewNode("apc-turret-root")
        self.apc_turret_np.setPos(0, 0.10, 1.02)
        make_box(self.apc_turret_np, "apc-lmg-mount", size=(0.78, 0.62, 0.34), color=(0.08, 0.10, 0.09, 1), pos=(0, 0, 0))
        self.apc_barrel_np = make_box(self.apc_turret_np, "apc-lmg-barrel", size=(0.07, 1.35, 0.07), color=(0.035, 0.04, 0.035, 1), pos=(0, 0.94, 0.02))
        make_box(self.apc_turret_np, "apc-lmg-muzzle", size=(0.10, 0.14, 0.10), color=(0.01, 0.01, 0.01, 1), pos=(0, 1.64, 0.02))

        for sx in [-1, 1]:
            make_box(root, "apc-side-armour", size=(0.36, 4.05, 0.34), color=(0.045, 0.050, 0.045, 1), pos=(sx * 1.36, 0, 0.20))
            for wy in [-1.38, -0.70, 0.0, 0.70, 1.38]:
                make_sphere(self, root, "apc-wheel", Point3(sx * 1.38, wy, 0.20), (0.20, 0.075, 0.20), (0.018, 0.018, 0.018, 1))

        make_box(root, "friendly-apc-marker", size=(0.42, 0.42, 0.10), color=(0.05, 0.75, 1.0, 1), pos=(0, -0.45, 1.40))
        # V10 detail pass: storage boxes, vision blocks, antenna and side doors.
        for sx in [-1, 1]:
            make_box(root, "apc-side-door", size=(0.08, 0.92, 0.42), color=(0.080, 0.100, 0.090, 1), pos=(sx * 1.56, 0.10, 0.68))
            make_box(root, "apc-rear-storage", size=(0.16, 0.55, 0.24), color=(0.055, 0.070, 0.060, 1), pos=(sx * 0.86, -2.12, 0.82))
        for px in [-0.62, -0.22, 0.22, 0.62]:
            make_box(root, "apc-vision-block", size=(0.22, 0.08, 0.11), color=(0.02, 0.04, 0.035, 1), pos=(px, 2.35, 0.94))
        make_box(root, "apc-antenna", size=(0.025, 0.025, 0.80), color=(0.01, 0.012, 0.010, 1), pos=(-0.92, -1.30, 1.42))
        self.update_apc_visual()

    def update_apc_visual(self):
        if self.apc_np is None:
            return
        ground = self.terrain_height(self.apc_pos.x, self.apc_pos.y)
        self.apc_pos.z = ground + 0.48
        self.apc_np.setPos(self.apc_pos)
        self.apc_np.setH(self.apc_h)
        if self.apc_turret_np is not None:
            self.apc_turret_np.setH(self.yaw - self.apc_h)
            self.apc_turret_np.setP(max(-8.0, min(14.0, -self.pitch * 0.18)))

    def distance_to_apc(self):
        dx = self.player_pos.x - self.apc_pos.x
        dy = self.player_pos.y - self.apc_pos.y
        return math.sqrt(dx * dx + dy * dy)

    def update_vehicle_visual(self):
        if self.vehicle_np is None:
            return
        ground = self.terrain_height(self.vehicle_pos.x, self.vehicle_pos.y)
        self.vehicle_pos.z = ground + 0.55
        self.vehicle_np.setPos(self.vehicle_pos)
        self.vehicle_np.setH(self.vehicle_h)
        if self.vehicle_turret_np is not None:
            self.vehicle_turret_np.setH(self.yaw - self.vehicle_h)
            self.vehicle_turret_np.setP(max(-7.0, min(12.0, -self.pitch * 0.18)))

    def distance_to_vehicle(self):
        dx = self.player_pos.x - self.vehicle_pos.x
        dy = self.player_pos.y - self.vehicle_pos.y
        return math.sqrt(dx * dx + dy * dy)

    def handle_interact(self):
        """E key: leave/enter pillbox or enter/exit the player tank."""
        if self.paused:
            return
        if self.mortar_targeting:
            self.cancel_mortar_targeting()
            return
        if self.in_vehicle:
            self.exit_vehicle()
            return
        tank_dist = self.distance_to_vehicle()
        apc_dist = self.distance_to_apc()
        if min(tank_dist, apc_dist) < 4.2:
            chosen = "apc" if apc_dist < tank_dist else "tank"
            if chosen == "tank" and self.vehicle_health <= 0:
                self.set_status("TANK DESTROYED", 1.2)
                return
            if chosen == "apc" and self.apc_health <= 0:
                self.set_status("APC DESTROYED", 1.2)
                return
            self.enter_vehicle(chosen)
            return
        if self.in_pillbox:
            self.exit_pillbox()
            return
        # Re-enter pillbox when close to its wall/centre.
        dx = self.player_pos.x - self.pillbox_pos.x
        dy = self.player_pos.y - self.pillbox_pos.y
        if math.sqrt(dx * dx + dy * dy) < self.pillbox_radius + 2.2:
            self.enter_pillbox()
        else:
            self.set_status("MOVE NEAR TANK OR PILLBOX, THEN PRESS E", 1.4)

    def exit_pillbox(self):
        """Leave through the real rear doorway instead of teleporting through a wall."""
        door_x = self.pillbox_pos.x
        door_y = self.pillbox_pos.y - (self.pillbox_radius + 1.65)
        self.player_pos = Point3(door_x, door_y, self.terrain_height(door_x, door_y) + self.eye_height)
        self.player_vel = Vec3(0, 0, 0)
        self.in_pillbox = False
        # Face roughly out of the doorway so the exit direction is obvious.
        self.yaw = 180.0
        self.pitch = min(self.pitch, 10.0)
        self.set_status("LEFT THROUGH REAR DOOR - PRESS E NEAR DOOR TO RE-ENTER", 1.9)

    def enter_pillbox(self):
        self.in_pillbox = True
        self.player_pos = Point3(self.pillbox_pos.x, self.pillbox_pos.y, self.terrain_height(self.pillbox_pos.x, self.pillbox_pos.y) + self.eye_height)
        self.player_vel = Vec3(0, 0, 0)
        self.set_status("BACK INSIDE PILLBOX", 1.2)

    def enter_vehicle(self, kind="tank"):
        self.in_vehicle = True
        self.in_pillbox = False
        self.current_vehicle_kind = kind
        self.reloading = False
        self.scoped = False
        self.cam.node().getLens().setFov(68)
        self.mouse_sensitivity = 0.08
        if kind == "apc":
            self.yaw = self.apc_h
            self.pitch = -4.0
            self.set_status("APC ENTERED: WASD DRIVE, MOUSE AIM, HOLD SPACE FOR 200-RD LMG, E EXIT", 2.6)
        else:
            self.yaw = self.vehicle_h
            self.pitch = -4.0
            self.set_status("TANK ENTERED: WASD DRIVE, MOUSE AIM, CLICK CANNON, E EXIT", 2.2)

    def exit_vehicle(self):
        kind = self.current_vehicle_kind or "tank"
        self.in_vehicle = False
        self.current_vehicle_kind = None
        self.cam.node().getLens().setFov(self.normal_fov)
        self.mouse_sensitivity = 0.12
        if kind == "apc":
            h = self.apc_h
            pos = self.apc_pos
            self.apc_speed = 0.0
        else:
            h = self.vehicle_h
            pos = self.vehicle_pos
            self.vehicle_speed = 0.0
        side = Vec3(math.cos(math.radians(h)), -math.sin(math.radians(h)), 0)
        x = pos.x + side.x * 2.7
        y = pos.y + side.y * 2.7
        self.player_pos = Point3(x, y, self.terrain_height(x, y) + self.eye_height)
        self.player_vel = Vec3(0, 0, 0)
        self.set_status("EXITED APC" if kind == "apc" else "EXITED TANK", 1.2)

    def update_vehicle(self, dt):
        """Drive the selected vehicle. Tank uses cannon; APC uses Space-operated LMG."""
        if self.current_vehicle_kind == "apc":
            self.update_apc_vehicle(dt)
            return

        turn = 0.0
        if self.keys["a"]:
            turn += 1.0
        if self.keys["d"]:
            turn -= 1.0

        throttle = 0.0
        if self.keys["w"]:
            throttle += 1.0
        if self.keys["s"]:
            throttle -= 0.65

        # Heavy vehicle behaviour: slow acceleration, slower reverse, broad turning.
        target_speed = throttle * (9.2 if throttle > 0 else 5.2)
        accel = 7.5 if throttle != 0 else 4.2
        self.vehicle_speed += (target_speed - self.vehicle_speed) * min(1.0, accel * dt)

        turn_rate = 58.0 * (0.35 + min(1.0, abs(self.vehicle_speed) / 7.5))
        self.vehicle_h += turn * turn_rate * dt

        h_rad = math.radians(self.vehicle_h)
        forward = Vec3(math.sin(h_rad), math.cos(h_rad), 0)
        self.vehicle_pos.x += forward.x * self.vehicle_speed * dt
        self.vehicle_pos.y += forward.y * self.vehicle_speed * dt
        self.vehicle_pos.x = max(-52, min(52, self.vehicle_pos.x))
        self.vehicle_pos.y = max(-52, min(52, self.vehicle_pos.y))

        self.update_vehicle_visual()
        self.player_pos = Point3(self.vehicle_pos.x, self.vehicle_pos.y, self.vehicle_pos.z + 1.25)

        # Camera sits in/above the turret. It aims with mouse, not hull direction.
        cam_pos = Point3(self.vehicle_pos.x, self.vehicle_pos.y, self.vehicle_pos.z + 1.65)
        self.camera.setPos(cam_pos)
        self.camera.setHpr(self.yaw, self.pitch, 0)

    def update_apc_vehicle(self, dt):
        turn = 0.0
        if self.keys["a"]:
            turn += 1.0
        if self.keys["d"]:
            turn -= 1.0

        throttle = 0.0
        if self.keys["w"]:
            throttle += 1.0
        if self.keys["s"]:
            throttle -= 0.70

        target_speed = throttle * (11.2 if throttle > 0 else 6.0)
        accel = 9.5 if throttle != 0 else 5.0
        self.apc_speed += (target_speed - self.apc_speed) * min(1.0, accel * dt)

        turn_rate = 72.0 * (0.35 + min(1.0, abs(self.apc_speed) / 8.5))
        self.apc_h += turn * turn_rate * dt

        h_rad = math.radians(self.apc_h)
        forward = Vec3(math.sin(h_rad), math.cos(h_rad), 0)
        self.apc_pos.x += forward.x * self.apc_speed * dt
        self.apc_pos.y += forward.y * self.apc_speed * dt
        self.apc_pos.x = max(-52, min(52, self.apc_pos.x))
        self.apc_pos.y = max(-52, min(52, self.apc_pos.y))

        self.update_apc_visual()
        self.player_pos = Point3(self.apc_pos.x, self.apc_pos.y, self.apc_pos.z + 1.22)
        cam_pos = Point3(self.apc_pos.x, self.apc_pos.y, self.apc_pos.z + 1.55)
        self.camera.setPos(cam_pos)
        self.camera.setHpr(self.yaw, self.pitch, 0)

    def reload_apc_lmg(self):
        if self.apc_lmg_reloading or self.apc_lmg_mag >= self.apc_lmg_weapon.mag_size or self.apc_lmg_reserve <= 0:
            return
        self.apc_lmg_reloading = True
        self.apc_lmg_reload_timer = self.apc_lmg_weapon.reload_time
        self.set_status("APC LMG RELOADING", 1.0)

    def finish_apc_reload(self):
        needed = self.apc_lmg_weapon.mag_size - self.apc_lmg_mag
        taken = min(needed, self.apc_lmg_reserve)
        self.apc_lmg_mag += taken
        self.apc_lmg_reserve -= taken
        self.apc_lmg_reloading = False
        self.apc_lmg_reload_timer = 0.0

    def fire_apc_lmg(self):
        if self.paused or self.mortar_targeting or not self.in_vehicle or self.current_vehicle_kind != "apc":
            return
        if self.apc_lmg_reloading:
            return
        if self.apc_lmg_mag <= 0:
            self.reload_apc_lmg()
            return
        if self.apc_lmg_fire_timer > 0:
            return

        weapon = self.apc_lmg_weapon
        self.apc_lmg_mag -= 1
        self.apc_lmg_fire_timer = 60.0 / weapon.rpm

        quat = self.camera.getQuat(self.render)
        forward = quat.getForward()
        right = quat.getRight()
        up = quat.getUp()
        spread = math.radians(weapon.spread_deg)
        direction = forward + right * random.gauss(0, spread) + up * random.gauss(0, spread)
        direction.normalize()
        start = self.camera.getPos(self.render) + direction * 1.85
        velocity = direction * weapon.muzzle_velocity
        projectile = make_projectile_visual(self, self.render, weapon, start, direction)
        self.bullets.append(Bullet(start, velocity, weapon, projectile))

    def switch_tank_ammo(self):
        """Z key: switch between HE and AP tank rounds."""
        if self.paused or self.mortar_targeting:
            return
        self.tank_ammo_index = (self.tank_ammo_index + 1) % len(self.tank_ammo_types)
        self.tank_weapon = self.tank_ammo_types[self.tank_ammo_index]
        role = "EXPLOSIVE / INFANTRY" if self.tank_weapon.name == "Tank HE" else "ARMOUR PIERCING / ANTI-TANK"
        self.set_status(f"LOADED {self.tank_weapon.name} - {role}", 1.4)

    def fire_vehicle_cannon(self):
        if self.current_vehicle_kind != "tank":
            return
        if self.vehicle_fire_timer > 0 or self.paused or self.mortar_targeting:
            return
        weapon = self.tank_ammo_types[self.tank_ammo_index]
        self.tank_weapon = weapon
        direction = self.camera.getQuat(self.render).getForward()
        direction.normalize()
        start = self.camera.getPos(self.render) + direction * 2.2
        velocity = direction * weapon.muzzle_velocity
        projectile = make_projectile_visual(self, self.render, weapon, start, direction)
        self.bullets.append(Bullet(start, velocity, weapon, projectile))
        self.vehicle_fire_timer = weapon.reload_time
        # HE has a larger recoil impulse; AP is higher-velocity but less blast/knockback.
        self.vehicle_speed -= 0.75 if weapon.name == "Tank HE" else 0.38
        self.set_status(f"{weapon.name} FIRED", 0.7)

    def setup_lighting(self):
        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.48, 0.48, 0.48, 1))
        self.render.setLight(self.render.attachNewNode(ambient))

        sun = DirectionalLight("sun")
        sun.setColor(Vec4(0.95, 0.92, 0.82, 1))

        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(-35, -55, 0)
        self.render.setLight(sun_np)

    # ----------------------------
    # HUD
    # ----------------------------

    def create_hud(self):
        # Top-left tactical info.
        self.hud_health = OnscreenText(text="", pos=(-1.28, 0.90), scale=0.055, align=TextNode.ALeft, fg=(1, 1, 1, 1), mayChange=True)
        self.hud_pillbox = OnscreenText(text="", pos=(-1.28, 0.82), scale=0.052, align=TextNode.ALeft, fg=(0.95, 0.95, 0.95, 1), mayChange=True)
        self.hud_info = OnscreenText(text="", pos=(-1.28, 0.74), scale=0.04, align=TextNode.ALeft, fg=(0.95, 0.95, 0.95, 1), mayChange=True)

        # V5: larger bottom-left status panel for health, money, wave state, and abilities.
        self.hud_bottom_panel = OnscreenText(text="", pos=(-1.30, -0.58), scale=0.052, align=TextNode.ALeft, fg=(0.96, 1.0, 0.90, 1), mayChange=True)
        self.hud_ammo = OnscreenText(text="", pos=(1.27, -0.88), scale=0.058, align=TextNode.ARight, fg=(1, 1, 1, 1), mayChange=True)
        self.hud_weapons = OnscreenText(text="", pos=(0, -0.93), scale=0.044, align=TextNode.ACenter, fg=(0.95, 0.95, 0.95, 1), mayChange=True)
        self.crosshair_np = None
        self.hud_hit = OnscreenText(text="", pos=(0, -0.13), scale=0.07, align=TextNode.ACenter, fg=(1, 0.90, 0.20, 1), mayChange=True)
        self.hud_warning = OnscreenText(text="", pos=(0, 0.82), scale=0.06, align=TextNode.ACenter, fg=(1, 0.25, 0.2, 1), mayChange=True)
        self.hud_status = OnscreenText(text="", pos=(0, 0.745), scale=0.043, align=TextNode.ACenter, fg=(0.72, 1.0, 0.72, 1), mayChange=True)
        self.scope_label = OnscreenText(text="", pos=(0, 0.67), scale=0.045, align=TextNode.ACenter, fg=(0.75, 1.0, 0.75, 1), mayChange=True)

    def draw_crosshair(self, moving=False, firing=False):
        """Line-drawn crosshair: no special font characters required."""
        if self.crosshair_np is not None:
            self.crosshair_np.removeNode()
            self.crosshair_np = None

        root = self.aspect2d.attachNewNode("line-crosshair")
        self.crosshair_np = root

        if self.scoped:
            gap = 0.018
            length = 0.080
            thickness = 2
        else:
            gap = 0.026 if not moving else 0.040
            length = 0.095 if not firing else 0.115
            thickness = 3

        seg = LineSegs("crosshair-lines")
        seg.setThickness(thickness)
        seg.setColor(0.72, 0.95, 1.0, 0.95)

        # Horizontal arms.
        seg.moveTo(-length, 0, 0)
        seg.drawTo(-gap, 0, 0)
        seg.moveTo(gap, 0, 0)
        seg.drawTo(length, 0, 0)

        # Vertical arms.
        seg.moveTo(0, 0, -length)
        seg.drawTo(0, 0, -gap)
        seg.moveTo(0, 0, gap)
        seg.drawTo(0, 0, length)

        # Tiny centre dot as a small square, not text.
        dot = 0.004
        seg.moveTo(-dot, 0, -dot)
        seg.drawTo(dot, 0, -dot)
        seg.drawTo(dot, 0, dot)
        seg.drawTo(-dot, 0, dot)
        seg.drawTo(-dot, 0, -dot)

        root.attachNewNode(seg.create())

    def draw_reload_ring(self, progress):
        if self.reload_ring_np is not None:
            self.reload_ring_np.removeNode()
            self.reload_ring_np = None

        if progress <= 0.0 or progress >= 1.0:
            return

        root = self.aspect2d.attachNewNode("reload-ring-root")
        radius = 0.095
        segments = 64

        red = LineSegs("reload-red")
        red.setThickness(4)
        red.setColor(1.0, 0.08, 0.06, 0.95)

        for i in range(segments + 1):
            angle = math.radians(-90 + 360 * i / segments)
            x = math.cos(angle) * radius
            z = math.sin(angle) * radius
            if i == 0:
                red.moveTo(x, 0, z)
            else:
                red.drawTo(x, 0, z)

        root.attachNewNode(red.create())

        green = LineSegs("reload-green")
        green.setThickness(5)
        green.setColor(0.05, 1.0, 0.20, 0.95)
        arc_segments = max(2, int(segments * progress))

        for i in range(arc_segments + 1):
            angle = math.radians(-90 - 360 * progress * i / arc_segments)
            x = math.cos(angle) * radius
            z = math.sin(angle) * radius
            if i == 0:
                green.moveTo(x, 0, z)
            else:
                green.drawTo(x, 0, z)

        root.attachNewNode(green.create())
        self.reload_ring_np = root

    def draw_scope_overlay(self):
        if self.scope_np is not None:
            self.scope_np.removeNode()
            self.scope_np = None

        if not self.scoped:
            return

        root = self.aspect2d.attachNewNode("scope-overlay")

        circle = LineSegs("scope-circle")
        circle.setThickness(3)
        circle.setColor(0.0, 0.0, 0.0, 0.9)
        radius = 0.57
        segments = 96

        for i in range(segments + 1):
            angle = math.tau * i / segments
            x = math.cos(angle) * radius
            z = math.sin(angle) * radius
            if i == 0:
                circle.moveTo(x, 0, z)
            else:
                circle.drawTo(x, 0, z)

        root.attachNewNode(circle.create())

        lines = LineSegs("scope-lines")
        lines.setThickness(2)
        lines.setColor(0.0, 0.0, 0.0, 0.9)
        lines.moveTo(-radius, 0, 0)
        lines.drawTo(-0.06, 0, 0)
        lines.moveTo(0.06, 0, 0)
        lines.drawTo(radius, 0, 0)
        lines.moveTo(0, 0, -radius)
        lines.drawTo(0, 0, -0.06)
        lines.moveTo(0, 0, 0.06)
        lines.drawTo(0, 0, radius)
        root.attachNewNode(lines.create())

        self.scope_np = root

    def update_hud(self):
        w = self.weapons[self.current_weapon]
        mag = self.magazines[self.current_weapon]
        reserve = self.reserves[self.current_weapon]

        if self.in_vehicle and self.current_vehicle_kind == "tank":
            self.hud_health.setText(f"Tank: {max(0, int(self.vehicle_health))}/{self.vehicle_max_health}")
            player_line = f"TANK     {max(0, int(self.vehicle_health))}/{self.vehicle_max_health}"
        elif self.in_vehicle and self.current_vehicle_kind == "apc":
            self.hud_health.setText(f"APC: {max(0, int(self.apc_health))}/{self.apc_max_health}")
            player_line = f"APC      {max(0, int(self.apc_health))}/{self.apc_max_health}"
        elif self.pillbox_health > 0:
            self.hud_health.setText(f"Player: PROTECTED / {self.health}")
            player_line = f"PLAYER   ARMOURED / {self.health}"
        else:
            self.hud_health.setText(f"Player: {max(0, int(self.health))}/100")
            player_line = f"PLAYER   {max(0, int(self.health))}/100"

        self.hud_pillbox.setText(f"Pillbox: {max(0, int(self.pillbox_health))}/{self.pillbox_max_health}")
        mortar_text = "READY" if self.mortar_cooldown <= 0 else f"{self.mortar_cooldown:.1f}s"
        if self.in_vehicle:
            mode_line = "MODE     APC" if self.current_vehicle_kind == "apc" else "MODE     TANK"
        else:
            mode_line = "MODE     PILLBOX" if self.in_pillbox else "MODE     OUTSIDE"
        tank_text = "READY" if self.vehicle_fire_timer <= 0 else f"{self.vehicle_fire_timer:.1f}s"
        if self.wave_break:
            wave_line = f"BREAK - Wave {self.next_wave_number} in {self.wave_break_timer:.1f}s"
        else:
            wave_line = f"Wave {self.wave}: {self.current_wave_name()}"

        self.hud_info.setText(
            f"{wave_line}   Kills {self.wave_kills}/{self.wave_goal}   Enemies {len(self.enemies)}"
        )
        if self.in_vehicle and self.current_vehicle_kind == "tank":
            tank_weapon = self.tank_ammo_types[self.tank_ammo_index]
            self.hud_ammo.setText(f"{tank_weapon.name}   {'READY' if self.vehicle_fire_timer <= 0 else f'{self.vehicle_fire_timer:.1f}s'}   Z SWITCH")
        elif self.in_vehicle and self.current_vehicle_kind == "apc":
            reload_note = " RELOADING" if self.apc_lmg_reloading else ""
            self.hud_ammo.setText(f"APC LMG   {self.apc_lmg_mag}/{self.apc_lmg_reserve}{reload_note}")
        else:
            self.hud_ammo.setText(f"{w.name}   {mag}/{reserve}")

        bottom = (
            f"{player_line}\n"
            f"PILLBOX  {max(0, int(self.pillbox_health))}/{self.pillbox_max_health}\n"
            f"MONEY    ${self.money}\n"
            f"WEAPON   {w.name}  {mag}/{reserve}\n"
            f"TANK AMMO Z = {self.tank_weapon.name}\n"
            f"GRENADES G = {self.grenade_count}\n"
            f"MORTAR   ${self.mortar_cost}  {mortar_text}\n"
            f"REPAIR   F = ${self.repair_cost}\n"
        )
        if self.mortar_targeting:
            bottom += "MORTAR TARGETING: arrow keys move, X confirm, C cancel\n"
        if self.wave_break:
            bottom += f"NEXT WAVE IN {self.wave_break_timer:.1f}s\n"
        self.hud_bottom_panel.setText(bottom)

        slots = []
        for i, weapon in enumerate(self.weapons):
            if i == self.current_weapon:
                slots.append(f"[{i + 1}:{weapon.name}]")
            else:
                slots.append(f" {i + 1}:{weapon.name} ")
        self.hud_weapons.setText("   ".join(slots))

        self.scope_label.setText("SCOPE" if self.scoped else "")

        if self.status_timer > 0:
            self.hud_status.setText(self.status_message)
        else:
            self.hud_status.setText("")

        if self.paused:
            self.hud_warning.setText("PAUSED")
        elif self.mortar_targeting:
            self.hud_warning.setText("MORTAR TARGETING")
        elif self.wave_break:
            self.hud_warning.setText(f"WAVE BREAK - NEXT WAVE IN {self.wave_break_timer:.1f}s")
        elif self.reloading:
            self.hud_warning.setText("RELOADING")
            progress = 1.0 - max(0.0, self.reload_timer / max(0.001, self.reload_total))
            self.draw_reload_ring(progress)
        else:
            self.draw_reload_ring(0.0)
            if mag == 0:
                self.hud_warning.setText("PRESS R TO RELOAD")
            elif self.pillbox_health <= 70 and self.pillbox_health > 0:
                self.hud_warning.setText("PILLBOX DAMAGED")
            elif self.pillbox_health <= 0:
                self.hud_warning.setText("PILLBOX DESTROYED - PLAYER VULNERABLE")
            else:
                self.hud_warning.setText("")

        self.hud_hit.setText("X" if self.hit_marker_timer > 0 else "")

    # ----------------------------
    # Minimap / radar
    # ----------------------------

    def draw_minimap(self):
        if self.minimap_np is not None:
            self.minimap_np.removeNode()
            self.minimap_np = None

        root = self.aspect2d.attachNewNode("radar-minimap")
        self.minimap_np = root

        cx = 1.05
        cz = 0.68
        radius = 0.22
        world_range = 55.0

        def world_to_radar(wx, wy):
            dx = wx - self.player_pos.x
            dy = wy - self.player_pos.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > world_range:
                return None
            rx = cx + (dx / world_range) * radius
            rz = cz + (dy / world_range) * radius
            if (rx - cx) ** 2 + (rz - cz) ** 2 > radius ** 2:
                return None
            return rx, rz

        def add_circle(name, x, z, r, color, thickness=1, segments=32):
            seg = LineSegs(name)
            seg.setThickness(thickness)
            seg.setColor(*color)
            for i in range(segments + 1):
                a = math.tau * i / segments
                px = x + math.cos(a) * r
                pz = z + math.sin(a) * r
                if i == 0:
                    seg.moveTo(px, 0, pz)
                else:
                    seg.drawTo(px, 0, pz)
            root.attachNewNode(seg.create())

        def add_line(name, points, color, thickness=1):
            if len(points) < 2:
                return
            seg = LineSegs(name)
            seg.setThickness(thickness)
            seg.setColor(*color)
            seg.moveTo(points[0][0], 0, points[0][1])
            for p in points[1:]:
                seg.drawTo(p[0], 0, p[1])
            root.attachNewNode(seg.create())

        add_circle("radar-outer", cx, cz, radius, (0.65, 0.85, 0.95, 0.95), 2, 72)
        for frac in [0.25, 0.50, 0.75]:
            add_circle("radar-ring", cx, cz, radius * frac, (0.35, 0.55, 0.65, 0.7), 1, 64)

        add_line("radar-x-axis", [(cx - radius, cz), (cx + radius, cz)], (0.25, 0.45, 0.55, 0.65), 1)
        add_line("radar-y-axis", [(cx, cz - radius), (cx, cz + radius)], (0.25, 0.45, 0.55, 0.65), 1)

        terrain_seg = LineSegs("radar-terrain")
        terrain_seg.setThickness(2)
        terrain_seg.setColor(0.45, 0.33, 0.18, 0.70)

        for gx in range(-48, 49, 8):
            for gy in range(-48, 49, 8):
                wx = self.player_pos.x + gx
                wy = self.player_pos.y + gy
                if self.terrain_height(wx, wy) < -0.55:
                    p = world_to_radar(wx, wy)
                    if p is not None:
                        x, z = p
                        terrain_seg.moveTo(x - 0.003, 0, z)
                        terrain_seg.drawTo(x + 0.003, 0, z)

        root.attachNewNode(terrain_seg.create())

        for ox, oy in self.cover_positions:
            p = world_to_radar(ox, oy)
            if p is None:
                continue
            x, z = p
            sz = 0.006
            add_line("cover-box", [(x - sz, z - sz), (x + sz, z - sz), (x + sz, z + sz), (x - sz, z + sz), (x - sz, z - sz)], (0.62, 0.48, 0.25, 0.95), 1)

        pbox = world_to_radar(self.pillbox_pos.x, self.pillbox_pos.y)
        if pbox is not None:
            px, pz = pbox
            add_circle("pillbox-marker", px, pz, 0.018, (0.2, 0.65, 1.0, 1.0), 2, 24)

        # Friendly vehicles.
        tank_p = world_to_radar(self.vehicle_pos.x, self.vehicle_pos.y)
        if tank_p is not None:
            tx, tz = tank_p
            add_circle("friendly-tank-marker", tx, tz, 0.014, (0.05, 0.35, 1.0, 1.0), 2, 16)
        apc_p = world_to_radar(self.apc_pos.x, self.apc_pos.y)
        if apc_p is not None:
            ax, az = apc_p
            add_circle("friendly-apc-marker", ax, az, 0.014, (0.05, 0.85, 1.0, 1.0), 2, 16)

        for enemy in self.enemies:
            if not enemy.alive:
                continue
            ep = enemy.world_pos()
            p = world_to_radar(ep.x, ep.y)
            if p is not None:
                ex, ez = p
                add_circle("enemy-o", ex, ez, 0.0075, (1.0, 0.15, 0.08, 1.0), 2, 14)


        # V5 mortar targeting cursor shown directly on the radar.
        if self.mortar_targeting:
            tx = self.player_pos.x + self.mortar_target_rel.x
            ty = self.player_pos.y + self.mortar_target_rel.y
            target_p = world_to_radar(tx, ty)
            if target_p is not None:
                mx, mz = target_p
                add_circle("mortar-target-ring", mx, mz, 0.018, (1.0, 0.95, 0.12, 1.0), 3, 24)
                add_line("mortar-target-cross-a", [(mx - 0.028, mz), (mx + 0.028, mz)], (1.0, 0.95, 0.12, 1.0), 2)
                add_line("mortar-target-cross-b", [(mx, mz - 0.028), (mx, mz + 0.028)], (1.0, 0.95, 0.12, 1.0), 2)
                add_line("mortar-target-vector", [(cx, cz), (mx, mz)], (1.0, 0.95, 0.12, 0.65), 1)

        # Fixed radar direction: use -self.yaw to match mouse look.
        yaw_rad = math.radians(-self.yaw)
        fx = math.sin(yaw_rad)
        fz = math.cos(yaw_rad)
        rx = math.cos(yaw_rad)
        rz = -math.sin(yaw_rad)

        tip = (cx + fx * 0.022, cz + fz * 0.022)
        left = (cx - fx * 0.013 - rx * 0.012, cz - fz * 0.013 - rz * 0.012)
        right = (cx - fx * 0.013 + rx * 0.012, cz - fz * 0.013 + rz * 0.012)
        add_line("player-triangle", [tip, left, right, tip], (0.05, 1.0, 0.20, 1.0), 2)

        cone_len = radius * 0.72
        cone_angle = math.radians(28)
        left_angle = yaw_rad - cone_angle
        right_angle = yaw_rad + cone_angle
        left_end = (cx + math.sin(left_angle) * cone_len, cz + math.cos(left_angle) * cone_len)
        right_end = (cx + math.sin(right_angle) * cone_len, cz + math.cos(right_angle) * cone_len)
        add_line("view-v-left", [(cx, cz), left_end], (0.1, 1.0, 0.25, 0.85), 1)
        add_line("view-v-right", [(cx, cz), right_end], (0.1, 1.0, 0.25, 0.85), 1)

        OnscreenText(text="RADAR", parent=root, pos=(cx, cz - radius - 0.045), scale=0.025, align=TextNode.ACenter, fg=(0.65, 0.9, 1.0, 0.95), mayChange=False)

    # ----------------------------
    # Input
    # ----------------------------

    def bind_keys(self):
        for key in ["w", "a", "s", "d", "space", "shift", "arrow_left", "arrow_right", "arrow_up", "arrow_down"]:
            self.accept(key, self.set_key, [key, True])
            self.accept(key + "-up", self.set_key, [key, False])

        self.accept("mouse1", self.set_mouse, [True])
        self.accept("mouse1-up", self.set_mouse, [False])
        self.accept("r", self.handle_r_key)
        self.accept("q", self.handle_q_key)
        self.accept("c", self.handle_c_key)
        self.accept("m", self.toggle_mouse_lock)
        self.accept("e", self.handle_interact)
        self.accept("t", self.debug_test_explosion)
        self.accept("g", self.throw_grenade)
        self.accept("z", self.switch_tank_ammo)
        self.accept("f", self.buy_pillbox_repair)
        self.accept("x", self.toggle_or_confirm_mortar_target)
        self.accept("enter", self.confirm_mortar_target)
        self.accept("1", self.switch_weapon, [0])
        self.accept("2", self.switch_weapon, [1])
        self.accept("3", self.switch_weapon, [2])
        self.accept("4", self.switch_weapon, [3])
        self.accept("escape", self.toggle_pause)

    def set_key(self, key, value):
        self.keys[key] = value

    def set_mouse(self, value):
        self.mouse_down = value

    def set_mouse_lock(self, locked):
        self.mouse_locked = locked
        props = WindowProperties()
        props.setCursorHidden(locked)
        self.win.requestProperties(props)
        if locked:
            self.win.movePointer(0, self.center_x, self.center_y)

    def toggle_mouse_lock(self):
        self.set_mouse_lock(not self.mouse_locked)
        self.set_status("MOUSE LOCKED" if self.mouse_locked else "MOUSE RELEASED", 1.0)

    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.set_mouse_lock(False)
            self.mortar_targeting = False
            self.show_pause_menu()
        else:
            self.hide_pause_menu()
            self.set_mouse_lock(True)

    def show_pause_menu(self):
        self.hide_pause_menu()
        root = self.aspect2d.attachNewNode("pause-menu")
        self.pause_menu_np = root
        seg = LineSegs("pause-box")
        seg.setThickness(3)
        seg.setColor(0.85, 0.95, 1.0, 0.95)
        x0, x1 = -0.52, 0.52
        z0, z1 = -0.34, 0.34
        seg.moveTo(x0, 0, z0); seg.drawTo(x1, 0, z0); seg.drawTo(x1, 0, z1); seg.drawTo(x0, 0, z1); seg.drawTo(x0, 0, z0)
        root.attachNewNode(seg.create())
        OnscreenText(text="PAUSED", parent=root, pos=(0, 0.23), scale=0.075, align=TextNode.ACenter, fg=(1,1,1,1), mayChange=False)
        OnscreenText(text="C  CONTINUE", parent=root, pos=(0, 0.10), scale=0.048, align=TextNode.ACenter, fg=(0.75,1.0,0.75,1), mayChange=False)
        OnscreenText(text="R  RESTART", parent=root, pos=(0, 0.00), scale=0.048, align=TextNode.ACenter, fg=(1.0,0.95,0.65,1), mayChange=False)
        OnscreenText(text="M  RELEASE / LOCK MOUSE", parent=root, pos=(0, -0.10), scale=0.043, align=TextNode.ACenter, fg=(0.75,0.95,1.0,1), mayChange=False)
        OnscreenText(text="Q  QUIT", parent=root, pos=(0, -0.21), scale=0.048, align=TextNode.ACenter, fg=(1.0,0.55,0.55,1), mayChange=False)

    def hide_pause_menu(self):
        if self.pause_menu_np is not None:
            self.pause_menu_np.removeNode()
            self.pause_menu_np = None

    def restart_game(self):
        os.execv(sys.executable, [sys.executable] + sys.argv)

    def handle_q_key(self):
        if self.paused:
            self.userExit()
        else:
            self.toggle_scope()

    def handle_r_key(self):
        if self.paused:
            self.restart_game()
        elif self.in_vehicle and self.current_vehicle_kind == "apc":
            self.reload_apc_lmg()
        else:
            self.start_reload()

    def handle_c_key(self):
        if self.paused:
            self.toggle_pause()
        elif self.mortar_targeting:
            self.cancel_mortar_targeting()



    def set_status(self, message, duration=1.7):
        self.status_message = message
        self.status_timer = duration

    def buy_pillbox_repair(self):
        if self.pillbox_health >= self.pillbox_max_health:
            self.set_status("PILLBOX ALREADY FULL", 1.2)
            return
        if self.money < self.repair_cost:
            self.set_status(f"NEED ${self.repair_cost} FOR REPAIR", 1.2)
            return
        self.money -= self.repair_cost
        self.pillbox_health = min(self.pillbox_max_health, self.pillbox_health + 95)
        self.set_status("PILLBOX REPAIRED +95", 1.6)

    def aim_ground_point(self, max_distance=46.0):
        start = self.camera.getPos(self.render)
        direction = self.camera.getQuat(self.render).getForward()
        best = Point3(start + direction * max_distance)
        for i in range(8, 95):
            d = max_distance * i / 95.0
            p = Point3(start + direction * d)
            ground = self.terrain_height(p.x, p.y) + 0.08
            if p.z <= ground:
                return Point3(p.x, p.y, ground)
            best = p
        best.z = self.terrain_height(best.x, best.y) + 0.08
        return best

    def toggle_or_confirm_mortar_target(self):
        if self.paused:
            return
        if self.mortar_targeting:
            self.confirm_mortar_target()
            return
        if self.mortar_cooldown > 0:
            self.set_status(f"MORTAR RELOADING {self.mortar_cooldown:.1f}s", 1.2)
            return
        if self.money < self.mortar_cost:
            self.set_status(f"NEED ${self.mortar_cost} FOR MORTAR", 1.2)
            return
        # Start target roughly in front of the player on the radar.
        yaw_rad = math.radians(self.yaw)
        self.mortar_target_rel = Vec3(math.sin(yaw_rad) * 34.0, math.cos(yaw_rad) * 34.0, 0)
        self.mortar_targeting = True
        self.set_status("MORTAR TARGETING: ARROW KEYS MOVE, X/ENTER CONFIRM, C CANCEL", 2.5)

    def cancel_mortar_targeting(self):
        self.mortar_targeting = False
        self.set_status("MORTAR CANCELLED", 1.0)

    def update_mortar_targeting(self, dt):
        if not self.mortar_targeting:
            return
        speed = 32.0 * dt
        move = Vec3(0, 0, 0)
        if self.keys["arrow_left"]:
            move.x -= speed
        if self.keys["arrow_right"]:
            move.x += speed
        if self.keys["arrow_up"]:
            move.y += speed
        if self.keys["arrow_down"]:
            move.y -= speed
        self.mortar_target_rel += move
        dist = math.sqrt(self.mortar_target_rel.x ** 2 + self.mortar_target_rel.y ** 2)
        if dist > self.mortar_target_range:
            scale = self.mortar_target_range / max(0.001, dist)
            self.mortar_target_rel.x *= scale
            self.mortar_target_rel.y *= scale

    def confirm_mortar_target(self):
        if not self.mortar_targeting:
            return
        center = Point3(
            self.player_pos.x + self.mortar_target_rel.x,
            self.player_pos.y + self.mortar_target_rel.y,
            self.terrain_height(self.player_pos.x + self.mortar_target_rel.x, self.player_pos.y + self.mortar_target_rel.y) + 0.10,
        )
        self.mortar_targeting = False
        self.call_mortar_strike_at(center)

    def call_mortar_strike_at(self, center):
        if self.mortar_cooldown > 0:
            self.set_status(f"MORTAR RELOADING {self.mortar_cooldown:.1f}s", 1.2)
            return
        if self.money < self.mortar_cost:
            self.set_status(f"NEED ${self.mortar_cost} FOR MORTAR", 1.2)
            return

        self.money -= self.mortar_cost
        self.mortar_cooldown = 16.0
        self.set_status("TARGETED MORTAR STRIKE INBOUND", 2.0)

        offsets = [(0, 0), (-3.2, 1.7), (3.1, -1.4), (-1.7, -3.0), (2.2, 3.2)]
        rpg_weapon = self.weapons[3]

        for idx, (ox, oy) in enumerate(offsets):
            def strike_task(task, ox=ox, oy=oy):
                p = Point3(center.x + ox, center.y + oy, self.terrain_height(center.x + ox, center.y + oy) + 0.10)
                self.create_impact_effect(p, rpg_weapon)
                self.apply_splash_damage(p, rpg_weapon)
                return Task.done
            self.taskMgr.doMethodLater(0.18 * idx, strike_task, f"targeted-mortar-strike-{self.time}-{idx}")

    def debug_test_explosion(self):
        """Press T to spawn a clearly visible test explosion 10 m in front of the player."""
        forward = self.camera.getQuat(self.render).getForward()
        pos = self.camera.getPos(self.render) + forward * 10.0
        pos.z = max(pos.z, self.terrain_height(pos.x, pos.y) + 0.25)
        self.create_impact_effect(Point3(pos), self.weapons[3])

    def toggle_scope(self):
        self.scoped = not self.scoped

        if self.scoped:
            self.cam.node().getLens().setFov(self.scope_fov)
            self.mouse_sensitivity = 0.055
        else:
            self.cam.node().getLens().setFov(self.normal_fov)
            self.mouse_sensitivity = 0.12

        self.draw_scope_overlay()

    # ----------------------------
    # Movement
    # ----------------------------

    def update_mouse_look(self):
        if self.paused or self.mortar_targeting or not self.mouse_locked:
            return
        if not self.mouseWatcherNode.hasMouse():
            return

        pointer = self.win.getPointer(0)
        dx = pointer.getX() - self.center_x
        dy = pointer.getY() - self.center_y

        self.yaw -= dx * self.mouse_sensitivity
        self.pitch -= dy * self.mouse_sensitivity
        self.pitch = max(-82, min(82, self.pitch))

        self.win.movePointer(0, self.center_x, self.center_y)

    def update_player(self, dt):
        self.update_mouse_look()

        if self.in_vehicle:
            self.update_vehicle(dt)
            return

        if self.mortar_targeting:
            self.player_vel.x = 0
            self.player_vel.y = 0
            self.camera.setPos(self.player_pos)
            self.camera.setHpr(self.yaw, self.pitch, 0)
            return

        yaw_rad = math.radians(self.yaw)

        # V7 movement fix: WASD is now relative to the direction you are actually looking.
        # Panda3D's heading sign means camera-forward is -sin(H), cos(H).
        # Example: look left, press W -> move left in the world.
        forward = Vec3(-math.sin(yaw_rad), math.cos(yaw_rad), 0)
        right = Vec3(math.cos(yaw_rad), math.sin(yaw_rad), 0)

        wish = Vec3(0, 0, 0)
        if self.keys["w"]:
            wish += forward
        if self.keys["s"]:
            wish -= forward
        if self.keys["a"]:
            wish -= right
        if self.keys["d"]:
            wish += right

        if wish.length() > 0:
            wish.normalize()

        speed = 6.4 if self.keys["shift"] else 4.8
        if self.scoped:
            speed *= 0.55

        self.player_vel.x = wish.x * speed
        self.player_vel.y = wish.y * speed

        ground_z = self.terrain_height(self.player_pos.x, self.player_pos.y) + self.eye_height

        if self.player_pos.z <= ground_z + 0.03:
            self.grounded = True
            self.player_pos.z = ground_z
            self.player_vel.z = max(0, self.player_vel.z)
        else:
            self.grounded = False

        if self.keys["space"] and self.grounded:
            self.player_vel.z = 5.1
            self.grounded = False

        self.player_vel.z -= 9.81 * dt
        self.player_pos += self.player_vel * dt

        dx = self.player_pos.x - self.pillbox_pos.x
        dy = self.player_pos.y - self.pillbox_pos.y
        dist = math.sqrt(dx * dx + dy * dy)
        max_dist = self.pillbox_radius - 0.75

        # V7: rear doorway lets you physically walk out.
        # The normal circular wall still blocks you, but the rear opening does not.
        rear_door_width = 3.0
        in_rear_door_sector = (dy < -max_dist * 0.55 and abs(dx) < rear_door_width * 0.5)

        if self.in_pillbox and dist > max_dist:
            if in_rear_door_sector:
                self.in_pillbox = False
                self.set_status("EXITED PILLBOX THROUGH REAR DOOR", 1.3)
            else:
                scale = max_dist / max(0.001, dist)
                self.player_pos.x = self.pillbox_pos.x + dx * scale
                self.player_pos.y = self.pillbox_pos.y + dy * scale
        elif not self.in_pillbox:
            # Outside the bunker you can roam the battlefield, but stay inside the generated terrain.
            self.player_pos.x = max(-52, min(52, self.player_pos.x))
            self.player_pos.y = max(-52, min(52, self.player_pos.y))

            # Re-enter naturally by walking back through the rear door.
            if dist < max_dist - 0.15 and dy < 0 and abs(dx) < rear_door_width * 0.5:
                self.in_pillbox = True
                self.set_status("RE-ENTERED PILLBOX THROUGH REAR DOOR", 1.2)

        new_ground_z = self.terrain_height(self.player_pos.x, self.player_pos.y) + self.eye_height

        if self.player_pos.z < new_ground_z:
            self.player_pos.z = new_ground_z
            self.player_vel.z = 0
            self.grounded = True

        self.camera.setPos(self.player_pos)
        self.camera.setHpr(self.yaw, self.pitch, 0)

    # ----------------------------
    # Weapons
    # ----------------------------

    def switch_weapon(self, idx):
        if 0 <= idx < len(self.weapons):
            self.current_weapon = idx
            self.reloading = False
            self.reload_timer = 0.0
            self.reload_total = 0.0

    def throw_grenade(self):
        if self.paused or self.mortar_targeting or self.in_vehicle or self.game_over:
            return
        if self.grenade_count <= 0:
            self.set_status("NO GRENADES LEFT", 1.0)
            return
        if self.grenade_cooldown > 0:
            return

        weapon = self.grenade_weapon
        self.grenade_count -= 1
        self.grenade_cooldown = 1.15

        quat = self.camera.getQuat(self.render)
        forward = quat.getForward()
        up = quat.getUp()
        direction = forward + up * 0.35
        direction.normalize()
        start = self.camera.getPos(self.render) + forward * 0.75 + Vec3(0, 0, -0.10)
        velocity = direction * weapon.muzzle_velocity + Vec3(0, 0, 1.8)
        node = make_projectile_visual(self, self.render, weapon, start, direction)
        grenade = Bullet(start, velocity, weapon, node)
        grenade.ttl = 4.0
        grenade.fuse = 1.35
        grenade.source = "player_grenade"
        self.bullets.append(grenade)
        self.set_status("GRENADE OUT", 0.8)

    def start_reload(self):
        if self.reloading:
            return

        idx = self.current_weapon
        w = self.weapons[idx]

        if self.magazines[idx] >= w.mag_size:
            return
        if self.reserves[idx] <= 0:
            return

        self.reloading = True
        self.reload_timer = w.reload_time
        self.reload_total = w.reload_time

    def finish_reload(self):
        idx = self.current_weapon
        w = self.weapons[idx]
        needed = w.mag_size - self.magazines[idx]
        taken = min(needed, self.reserves[idx])
        self.magazines[idx] += taken
        self.reserves[idx] -= taken
        self.reloading = False
        self.reload_timer = 0.0
        self.reload_total = 0.0

    def try_fire(self):
        if self.in_vehicle:
            if self.current_vehicle_kind == "tank":
                self.fire_vehicle_cannon()
            return
        if self.game_over or self.reloading or self.paused or self.mortar_targeting:
            return

        idx = self.current_weapon
        w = self.weapons[idx]

        if self.magazines[idx] <= 0:
            self.start_reload()
            return

        if self.fire_timer > 0:
            return

        self.magazines[idx] -= 1
        self.fire_timer = 60.0 / w.rpm

        for _ in range(w.pellets):
            self.spawn_bullet(w)

    def spawn_bullet(self, weapon):
        cam_pos = self.camera.getPos(self.render)
        quat = self.camera.getQuat(self.render)
        forward = quat.getForward()
        right = quat.getRight()
        up = quat.getUp()

        spread_deg = weapon.spread_deg * (0.35 if self.scoped else 1.0)
        spread = math.radians(spread_deg)
        spread_x = random.gauss(0, spread)
        spread_y = random.gauss(0, spread)

        direction = forward + right * spread_x + up * spread_y
        direction.normalize()

        start = cam_pos + direction * 0.85
        velocity = direction * weapon.muzzle_velocity

        bullet_node = make_projectile_visual(self, self.render, weapon, start, direction)
        bullet = Bullet(start, velocity, weapon, bullet_node)
        self.bullets.append(bullet)

    # ----------------------------
    # Bullet physics and hits
    # ----------------------------

    def update_bullets(self, dt):
        remaining = []

        for bullet in self.bullets:
            old_pos = Point3(bullet.pos)
            bullet.vel.z -= 9.81 * bullet.gravity_scale * dt
            bullet.pos += bullet.vel * dt
            bullet.ttl -= dt
            bullet.node.setPos(bullet.pos)
            if bullet.vel.lengthSquared() > 0.0001:
                look_dir = Vec3(bullet.vel)
                look_dir.normalize()
                bullet.node.lookAt(Point3(bullet.pos) + look_dir)
            hit = False

            if bullet.pos.z <= self.terrain_height(bullet.pos.x, bullet.pos.y):
                if bullet.weapon.name == "Grenade" and bullet.bounces < 1 and bullet.vel.length() > 5.0:
                    bullet.pos.z = self.terrain_height(bullet.pos.x, bullet.pos.y) + 0.10
                    bullet.vel.z = abs(bullet.vel.z) * 0.32
                    bullet.vel.x *= 0.52
                    bullet.vel.y *= 0.52
                    bullet.bounces += 1
                    bullet.node.setPos(bullet.pos)
                else:
                    hit = True
                    self.create_impact_effect(bullet.pos, bullet.weapon)
                    if bullet.weapon.splash_radius > 0:
                        self.apply_splash_damage(bullet.pos, bullet.weapon)

            if not hit:
                hit = self.check_bullet_enemy_hit(old_pos, bullet.pos, bullet)

            if bullet.weapon.name == "Grenade" and bullet.fuse is not None and bullet.ttl <= bullet.fuse:
                # Fuse is counted from default 4.0 downwards. When remaining TTL is below fuse threshold, detonate.
                hit = True
                self.create_impact_effect(bullet.pos, bullet.weapon)
                self.apply_splash_damage(bullet.pos, bullet.weapon)

            if hit or bullet.ttl <= 0:
                bullet.node.removeNode()
            else:
                remaining.append(bullet)

        self.bullets = remaining

    def arm_enemy_tank_return_fire(self, enemy):
        """Wake an enemy tank after it has been shot.

        V9 fix: the previous return-fire could be hard to notice because the
        cooldown might still be several seconds. Now the first response is
        queued almost immediately after the tank is hit.
        """
        if not enemy.alive or enemy.enemy_type.name != "Tank":
            return
        enemy.retaliation_timer = max(enemy.retaliation_timer, 18.0)
        # First retaliatory shot is delayed enough to be fair, then 3-5 s between shots.
        enemy.return_fire_cooldown = min(enemy.return_fire_cooldown, 1.25)
        self.set_status("ENEMY TANK ANGERED - RETURN FIRE SOON", 1.5)

    def armored_damage_amount(self, enemy, weapon, base_damage, multiplier=1.0, is_splash=False):
        """Armour/penetration model for enemy vehicles.

        V10 tank model:
        - Tank Cannon: high penetration; turret/rear hits are critical.
        - RPG: direct hits are serious and normally 2-shot tanks.
        - RPG near-miss splash damages tanks only lightly.
        - Rifle/SMG/Shotgun/APC LMG are negligible against tanks.
        - Track hits do reduced damage but slow the tank temporarily.
        """
        raw = base_damage * multiplier

        if enemy.enemy_type.name == "Tank":
            if weapon.name == "Tank AP":
                # AP has high penetration and high velocity, but almost no useful splash.
                return raw * (1.45 if not is_splash else 0.06)
            if weapon.name in ("Tank HE", "Tank Cannon"):
                # HE can damage tanks, but it is less efficient than AP against armour.
                return raw * (0.82 if not is_splash else 0.34)
            if weapon.name == "RPG":
                return raw * (1.00 if not is_splash else 0.28)
            if weapon.name == "Grenade":
                return min(12.0, raw * 0.05)
            return min(2.0, raw * 0.012)

        if enemy.enemy_type.name == "APC":
            if weapon.name in ("Tank HE", "Tank AP", "Tank Cannon", "RPG", "Grenade"):
                return raw
            if weapon.name == "APC LMG":
                return raw * 0.35
            return raw * 0.55

        return raw

    def apply_weapon_damage_to_enemy(self, enemy, weapon, base_damage, multiplier=1.0, is_splash=False):
        if enemy.enemy_type.name == "Tank":
            self.arm_enemy_tank_return_fire(enemy)

        amount = self.armored_damage_amount(enemy, weapon, base_damage, multiplier, is_splash)
        if amount > 0 and enemy.alive:
            enemy.apply_damage(amount)
        return amount

    def classify_tank_hit(self, enemy, world_point):
        """Approximate local armour zones for a tank.

        Returns (zone_name, multiplier, hit_marker_label). It is deliberately simple,
        but it gives gameplay-relevant realism: turret/rear hits hurt, track/front hits
        are less effective, and track hits slow the vehicle.
        """
        s = enemy.enemy_type.size_scale
        local = enemy.root.getRelativePoint(self.render, Point3(world_point))
        ax = abs(local.x)
        y = local.y
        z = local.z

        if ax > 0.98 * s and -0.08 * s <= z <= 0.72 * s:
            enemy.track_damage_timer = max(enemy.track_damage_timer, 5.0)
            return "track", 0.42, "TRACK HIT - MOBILITY DAMAGE"

        if ax <= 0.78 * s and -0.60 * s <= y <= 0.78 * s and 0.86 * s <= z <= 1.58 * s:
            return "turret", 1.35, "TURRET CRITICAL"

        if y < -0.92 * s and 0.45 * s <= z <= 1.25 * s:
            return "rear", 1.20, "ENGINE/REAR CRITICAL"

        if y > 1.08 * s and 0.35 * s <= z <= 1.18 * s:
            return "front", 0.68, "FRONTAL ARMOUR HIT"

        return "hull", 1.00, "HULL PENETRATION"

    def check_bullet_enemy_hit(self, old_pos, new_pos, bullet):
        samples = 6

        for i in range(samples + 1):
            t = i / samples
            p = old_pos * (1 - t) + new_pos * t

            for enemy in list(self.enemies):
                if not enemy.alive:
                    continue

                ep = enemy.world_pos()
                s = enemy.enemy_type.size_scale
                rel = p - ep
                horizontal_dist = math.sqrt(rel.x * rel.x + rel.y * rel.y)

                if horizontal_dist > 0.65 * s:
                    continue

                z = rel.z

                # V9: dedicated large vehicle hit volumes. The old humanoid head/chest
                # zones made tank hits unreliable, which is why return fire often did not appear.
                if enemy.enemy_type.name == "Tank":
                    if horizontal_dist <= 2.35 * s and -0.20 * s <= z <= 1.75 * s:
                        zone, zone_mult, label = self.classify_tank_hit(enemy, p)
                        amount = self.apply_weapon_damage_to_enemy(enemy, bullet.weapon, bullet.damage, zone_mult, is_splash=False)
                        if bullet.vel.lengthSquared() > 0.0001:
                            impact_dir = Vec3(bullet.vel)
                            impact_dir.normalize()
                            if bullet.weapon.name == "Tank HE":
                                kb, up = 8.0, 1.2
                            elif bullet.weapon.name == "Tank AP":
                                kb, up = 2.2, 0.25
                            elif bullet.weapon.name in ("Tank Cannon", "RPG"):
                                kb, up = 7.0, 1.1
                            else:
                                kb, up = 0.0, 0.0
                            self.apply_knockback_to_enemy(enemy, p - impact_dir * 0.2, kb, upward=up, weapon=bullet.weapon)
                        self.show_hit_marker(label if amount > 8 else "ARMOUR SPARK")
                        if bullet.weapon.splash_radius > 0:
                            self.apply_splash_damage(p, bullet.weapon, direct_hit_enemy=enemy)
                            self.create_impact_effect(p, bullet.weapon)
                        return True
                    continue

                if enemy.enemy_type.name == "APC":
                    if horizontal_dist <= 1.75 * s and -0.15 * s <= z <= 1.50 * s:
                        self.apply_weapon_damage_to_enemy(enemy, bullet.weapon, bullet.damage, 1.0, is_splash=False)
                        if bullet.vel.lengthSquared() > 0.0001:
                            impact_dir = Vec3(bullet.vel)
                            impact_dir.normalize()
                            self.apply_knockback_to_enemy(enemy, p - impact_dir * 0.2, 4.0, upward=0.8, weapon=bullet.weapon)
                        self.show_hit_marker("VEHICLE HIT")
                        if bullet.weapon.splash_radius > 0:
                            self.apply_splash_damage(p, bullet.weapon, direct_hit_enemy=enemy)
                            self.create_impact_effect(p, bullet.weapon)
                        return True
                    continue

                if 1.55 * s <= z <= 2.05 * s and horizontal_dist <= 0.36 * s:
                    multiplier = 2.8
                    label = "HEAD CRITICAL"
                elif 0.82 * s <= z <= 1.55 * s and horizontal_dist <= 0.52 * s:
                    multiplier = 1.65
                    label = "CHEST CRITICAL"
                elif 0.05 * s <= z <= 0.85 * s and horizontal_dist <= 0.52 * s:
                    multiplier = 1.0
                    label = "HIT"
                else:
                    continue

                self.apply_weapon_damage_to_enemy(enemy, bullet.weapon, bullet.damage, multiplier, is_splash=False)
                if bullet.vel.lengthSquared() > 0.0001:
                    impact_dir = Vec3(bullet.vel)
                    impact_dir.normalize()
                    self.apply_knockback_to_enemy(enemy, p - impact_dir * 0.2, 3.2 if bullet.weapon.name != "RPG" else 12.5, upward=0.65 if bullet.weapon.name != "RPG" else 3.2, weapon=bullet.weapon)
                self.show_hit_marker(label)

                if bullet.weapon.splash_radius > 0:
                    self.apply_splash_damage(p, bullet.weapon, direct_hit_enemy=enemy)
                    self.create_impact_effect(p, bullet.weapon)

                return True

        return False

    def apply_knockback_to_enemy(self, enemy, origin, strength, upward=0.0, weapon=None):
        """Apply physical-feeling knockback to an enemy using persistent velocity.

        V8 armour rule: small arms do not knock back enemy tank mobs. Explosives
        and cannon shells can still shove them because those have blast impulse.
        """
        if not enemy.alive:
            return
        # V10 armour rule: enemy tanks ignore small-arms/APC LMG knockback.
        # Tank shells can shove them; RPGs give a small visible jolt only.
        if enemy.enemy_type.name == "Tank":
            if weapon is None or weapon.name not in ("Tank HE", "Tank AP", "Tank Cannon", "RPG"):
                return
            if weapon.name == "Tank AP":
                # AP penetrates, but transfers much less sideways blast impulse.
                strength *= 0.32
                upward *= 0.18
            elif weapon.name == "RPG":
                strength *= 0.28
                upward *= 0.25
        if enemy.enemy_type.name == "APC" and (weapon is None or weapon.splash_radius <= 0):
            strength *= 0.35
            upward *= 0.25

        ep = enemy.world_pos() + Vec3(0, 0, 0.85)
        direction = Vec3(ep.x - origin.x, ep.y - origin.y, 0)

        if direction.lengthSquared() < 0.0001:
            direction = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), 0)

        direction.normalize()
        enemy.knockback_vel += direction * strength
        enemy.knockback_vel.z += upward

        # Clamp so enemies react strongly but do not launch unrealistically far.
        max_horizontal = 15.5
        horizontal = Vec3(enemy.knockback_vel.x, enemy.knockback_vel.y, 0)
        if horizontal.length() > max_horizontal:
            horizontal.normalize()
            horizontal *= max_horizontal
            enemy.knockback_vel.x = horizontal.x
            enemy.knockback_vel.y = horizontal.y
        enemy.knockback_vel.z = min(enemy.knockback_vel.z, 8.5)

    def apply_splash_damage(self, pos, weapon, direct_hit_enemy=None):
        for enemy in list(self.enemies):
            if not enemy.alive:
                continue
            if direct_hit_enemy is not None and enemy is direct_hit_enemy:
                continue
            ep = enemy.world_pos() + Vec3(0, 0, 1.0)
            dist = (ep - pos).length()
            if dist <= weapon.splash_radius:
                falloff = max(0.0, 1.0 - dist / weapon.splash_radius)
                damage = weapon.splash_damage * max(0.25, falloff)
                self.apply_weapon_damage_to_enemy(enemy, weapon, damage, 1.0, is_splash=True)
                self.apply_knockback_to_enemy(enemy, pos, 16.5 * max(0.30, falloff), upward=6.2 * max(0.20, falloff), weapon=weapon)
                self.show_hit_marker("SPLASH")

    def create_impact_effect(self, pos, weapon):
        """
        RPG/explosive impact effect.
        This version uses a procedural half-dome mesh, not a static sphere.
        The dome, shock ring, flash, sparks, and smoke are all updated every frame
        by update_explosions(dt). Press E in-game to spawn a test explosion.
        """
        if weapon.splash_radius <= 0:
            return

        root = self.render.attachNewNode("SMALLER_DOME_NO_RING_V4")
        root.setPos(pos)

        # Visual radius is deliberately smaller than the damage radius.
        # The previous dome was too large visually even though the gameplay splash was fine.
        radius = max(2.6, weapon.splash_radius * 0.68)

        flash = make_sphere(
            self, root, "blast-flash-core", Point3(0, 0, 0.35),
            0.35, (1.0, 0.75, 0.10, 1.0)
        )
        flash.setTransparency(True)
        flash.setLightOff()

        dome = make_expanding_dome_mesh(
            root,
            name="REAL_VISIBLE_EXPANDING_HALF_DOME_MESH",
            color=(1.0, 0.42, 0.04, 0.62),
            rings=12,
            segments=64,
        )
        # Start as a compact, taller cap instead of a flat pancake.
        dome.setScale(0.10, 0.10, 0.16)
        dome.setPos(0, 0, 0.02)

        inner_dome = make_expanding_dome_mesh(
            root,
            name="INNER_WHITE_HOT_DOME_MESH",
            color=(1.0, 0.82, 0.22, 0.55),
            rings=8,
            segments=48,
        )
        inner_dome.setScale(0.07, 0.07, 0.12)
        inner_dome.setPos(0, 0, 0.04)

        # Removed the yellow shock ring entirely.
        # It looked unrealistic and distracted from the expanding dome.
        ring_root = None

        sparks = []
        for _ in range(54):
            spark = make_sphere(
                self, root, "gravity-spark", Point3(0, 0, random.uniform(0.22, 0.70)),
                random.uniform(0.035, 0.085),
                (1.0, random.uniform(0.38, 0.90), 0.04, 1.0)
            )
            spark.setLightOff()
            spark_dir = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.18, 1.15))
            if spark_dir.lengthSquared() > 0:
                spark_dir.normalize()
            sparks.append({
                "node": spark,
                "vel": spark_dir * random.uniform(6.5, 17.0),
                "age": 0.0,
                "life": random.uniform(0.40, 0.95),
                "base_scale": spark.getScale().x,
            })

        smoke = []
        for _ in range(22):
            puff = make_sphere(
                self, root, "expanding-smoke-puff", Point3(0, 0, random.uniform(0.12, 0.55)),
                random.uniform(0.12, 0.30),
                (0.16, 0.15, 0.13, 0.58)
            )
            puff.setTransparency(True)
            puff.setLightOff()
            smoke_dir = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.22, 1.0))
            if smoke_dir.lengthSquared() > 0:
                smoke_dir.normalize()
            smoke.append({
                "node": puff,
                "vel": smoke_dir * random.uniform(1.2, 4.0),
                "age": 0.0,
                "life": random.uniform(1.00, 1.95),
                "base_scale": puff.getScale().x,
            })

        self.active_explosions.append({
            "root": root,
            "flash": flash,
            "dome": dome,
            "inner_dome": inner_dome,
            "ring_root": ring_root,
            "sparks": sparks,
            "smoke": smoke,
            "age": 0.0,
            "life": 1.95,
            "radius": radius,
            "build_marker": "SMALLER_DOME_VISIBLE_HP_KNOCKBACK_FEATURES_V4",
        })

    def update_explosions(self, dt):
        remaining = []

        for effect in self.active_explosions:
            effect["age"] += dt
            age = effect["age"]
            life = effect["life"]
            t = min(1.0, age / life)
            radius = effect["radius"]

            # Obvious expansion: most growth happens in the first 0.85 s.
            # The Z scale is now much larger so the blast reads as a dome,
            # not a flat ground ripple.
            grow_t = min(1.0, age / 0.85)
            expansion = 1.0 - (1.0 - grow_t) ** 2.8
            fade = max(0.0, 1.0 - t)

            flash = effect["flash"]
            if not flash.isEmpty():
                flash_scale = max(0.01, radius * 0.42 * max(0.0, 1.0 - age / 0.30))
                flash.setScale(flash_scale)
                flash.setColor(1.0, 0.62, 0.05, max(0.0, 1.0 - age / 0.33))
                if age > 0.36:
                    flash.removeNode()

            dome = effect["dome"]
            if not dome.isEmpty():
                dome_radius = 0.10 + radius * 0.92 * expansion
                dome_height = 0.15 + radius * 0.86 * expansion
                dome.setScale(dome_radius, dome_radius, dome_height)
                dome.setColor(1.0, 0.34, 0.035, 0.60 * fade)
                if t >= 1.0:
                    dome.removeNode()

            inner_dome = effect["inner_dome"]
            if not inner_dome.isEmpty():
                inner_t = min(1.0, age / 0.55)
                inner_expansion = 1.0 - (1.0 - inner_t) ** 2.0
                inner_radius = 0.06 + radius * 0.28 * inner_expansion
                inner_height = 0.11 + radius * 0.42 * inner_expansion
                inner_alpha = max(0.0, 0.45 * (1.0 - inner_t))
                inner_dome.setScale(inner_radius, inner_radius, inner_height)
                inner_dome.setColor(1.0, 0.55, 0.10, inner_alpha)
                if inner_t >= 1.0:
                    inner_dome.removeNode()

            # No expanding yellow ring in V3.

            alive_sparks = []
            for spark in effect["sparks"]:
                spark["age"] += dt
                node = spark["node"]
                if spark["age"] >= spark["life"] or node.isEmpty():
                    if not node.isEmpty():
                        node.removeNode()
                    continue

                spark["vel"] += Vec3(0, 0, -9.81) * dt * 0.65
                node.setPos(node.getPos() + spark["vel"] * dt)
                st = spark["age"] / spark["life"]
                node.setScale(max(0.004, spark["base_scale"] * (1.0 - st)))
                node.setColor(1.0, 0.52 + 0.35 * (1.0 - st), 0.04, 1.0 - st)
                alive_sparks.append(spark)

            alive_smoke = []
            for puff in effect["smoke"]:
                puff["age"] += dt
                node = puff["node"]
                if puff["age"] >= puff["life"] or node.isEmpty():
                    if not node.isEmpty():
                        node.removeNode()
                    continue

                puff["vel"] *= max(0.0, 1.0 - 0.70 * dt)
                node.setPos(node.getPos() + puff["vel"] * dt)
                pt = puff["age"] / puff["life"]
                node.setScale(puff["base_scale"] * (1.0 + pt * 3.0))
                node.setColor(0.16, 0.15, 0.13, max(0.0, 0.58 * (1.0 - pt)))
                alive_smoke.append(puff)

            effect["sparks"] = alive_sparks
            effect["smoke"] = alive_smoke

            if age >= life and not alive_sparks and not alive_smoke:
                if not effect["root"].isEmpty():
                    effect["root"].removeNode()
            else:
                remaining.append(effect)

        self.active_explosions = remaining

    def show_hit_marker(self, label):
        self.hit_marker_timer = 0.13
        if "HEAD" in label:
            self.hud_hit.setFg((1, 0.25, 0.1, 1))
        elif "CHEST" in label:
            self.hud_hit.setFg((1, 0.85, 0.2, 1))
        elif "SPLASH" in label:
            self.hud_hit.setFg((1, 0.45, 0.1, 1))
        else:
            self.hud_hit.setFg((0.85, 0.95, 1, 1))

    # ----------------------------
    # Enemy spawning / waves
    # ----------------------------

    def current_wave_name(self):
        pattern = self.wave % 4
        if pattern == 1:
            return "Frontal Assault"
        if pattern == 2:
            return "Fast Runners"
        if pattern == 3:
            return "Heavy Push"
        return "Mixed Attack"

    def pick_enemy_type_for_wave(self):
        pattern = self.wave % 4
        if pattern == 1:
            return random.choices(self.enemy_types, weights=[0.43, 0.12, 0.37, 0.05, 0.03], k=1)[0]
        if pattern == 2:
            return random.choices(self.enemy_types, weights=[0.75, 0.04, 0.18, 0.02, 0.01], k=1)[0]
        if pattern == 3:
            return random.choices(self.enemy_types, weights=[0.14, 0.41, 0.20, 0.15, 0.10], k=1)[0]
        return random.choices(self.enemy_types, weights=[0.34, 0.20, 0.25, 0.12, 0.09], k=1)[0]

    def spawn_enemy(self):
        pattern = self.wave % 4

        if pattern == 1:
            x = random.uniform(-32, 32)
            y = random.uniform(28, 50)
        elif pattern == 2:
            side = random.choice([-1, 1])
            x = side * random.uniform(18, 42)
            y = random.uniform(8, 42)
        elif pattern == 3:
            x = random.uniform(-18, 18)
            y = random.uniform(26, 48)
        else:
            if random.random() < 0.7:
                x = random.uniform(-35, 35)
                y = random.uniform(24, 50)
            else:
                side = random.choice([-1, 1])
                x = side * random.uniform(28, 45)
                y = random.uniform(4, 34)

        z = self.terrain_height(x, y)
        enemy_type = self.pick_enemy_type_for_wave()
        enemy = Enemy(self, Point3(x, y, z), enemy_type, level=self.wave)
        self.enemies.append(enemy)

    def update_enemies(self, dt):
        if self.wave_break:
            return

        self.spawn_timer -= dt
        target_count = min(18, 4 + self.wave * 2)

        if self.spawn_timer <= 0 and len(self.enemies) < target_count and self.wave_kills < self.wave_goal:
            self.spawn_enemy()
            self.spawn_timer = max(0.62, 1.85 - self.wave * 0.055)

        for enemy in list(self.enemies):
            enemy.update(dt)

        self.enemies = [e for e in self.enemies if e.alive]

        if self.wave_kills >= self.wave_goal and len(self.enemies) == 0:
            self.start_wave_break()

    def start_wave_break(self):
        if self.wave_break:
            return
        self.wave_break = True
        self.wave_break_timer = self.wave_break_duration
        self.next_wave_number = self.wave + 1
        self.set_status(f"WAVE {self.wave} CLEARED - PREPARE FOR WAVE {self.next_wave_number}", 3.0)

        # Clear any targeting mode during the break.
        self.mortar_targeting = False

        # Small immediate reward at wave clear.
        self.health = min(self.max_health, self.health + 15)
        self.pillbox_health = min(self.pillbox_max_health, self.pillbox_health + 35)
        self.money += 45 + self.wave * 8

    def begin_next_wave(self):
        self.wave_break = False
        self.wave += 1
        self.wave_kills = 0
        self.wave_goal = 10 + self.wave * 3
        self.spawn_timer = 1.0
        self.set_status(f"WAVE {self.wave}: {self.current_wave_name()}", 2.5)

        # Larger start-of-wave resupply.
        self.health = min(self.max_health, self.health + 10)
        self.pillbox_health = min(self.pillbox_max_health, self.pillbox_health + 45)
        for i, w in enumerate(self.weapons):
            if w.name == "RPG":
                self.reserves[i] += 2
            else:
                self.reserves[i] += w.mag_size
        self.grenade_count = min(8, self.grenade_count + 2)

    def next_wave(self):
        # Backwards-compatible alias used by earlier versions.
        self.start_wave_break()

    def fire_enemy_tank_round(self, enemy):
        """Enemy tank return fire: only used after the player has shot the tank."""
        if not enemy.alive or self.game_over or self.paused:
            return

        weapon = self.enemy_tank_shell_weapon
        enemy_pos = enemy.world_pos()
        origin = Point3(enemy_pos.x, enemy_pos.y, enemy_pos.z + 1.35)

        # V9: return fire should actually aim at what the player is using.
        # If you are in your tank/APC, the shell targets that vehicle. If you are
        # outside, it targets you. Only if you are inside the pillbox does it hit the pillbox.
        target_kind = "pillbox"
        if self.in_vehicle and self.current_vehicle_kind == "tank":
            target = Point3(self.vehicle_pos.x, self.vehicle_pos.y, self.vehicle_pos.z + 0.75)
            target_kind = "friendly_tank"
        elif self.in_vehicle and self.current_vehicle_kind == "apc":
            target = Point3(self.apc_pos.x, self.apc_pos.y, self.apc_pos.z + 0.65)
            target_kind = "friendly_apc"
        elif not self.in_pillbox:
            target = Point3(self.player_pos.x, self.player_pos.y, self.player_pos.z - 0.35)
            target_kind = "player"
        elif self.pillbox_health > 0:
            target = Point3(self.pillbox_pos.x, self.pillbox_pos.y, self.pillbox_pos.z + 1.05)
            target_kind = "pillbox"
        else:
            target = Point3(self.player_pos.x, self.player_pos.y, self.player_pos.z - 0.35)
            target_kind = "player"

        direction = Vec3(target - origin)
        if direction.lengthSquared() < 0.001:
            direction = Vec3(0, -1, 0)
        direction.normalize()

        projectile = make_projectile_visual(self, self.render, weapon, origin, direction)
        distance = (target - origin).length()
        flight_time = max(0.35, min(1.15, distance / max(1.0, weapon.muzzle_velocity)))

        self.set_status("ENEMY TANK RETURN FIRE!", 1.0)

        def shell_task(task):
            t = min(1.0, task.time / flight_time)
            # Slight arc so the shot reads as a heavy shell rather than a laser.
            pos = origin * (1.0 - t) + target * t
            pos.z += math.sin(math.pi * t) * 1.1
            projectile.setPos(pos)
            if direction.lengthSquared() > 0:
                projectile.lookAt(pos + direction)
            if t >= 1.0:
                projectile.removeNode()
                impact = Point3(target)
                impact.z = max(impact.z, self.terrain_height(impact.x, impact.y) + 0.12)
                self.create_impact_effect(impact, weapon)
                if target_kind == "friendly_tank":
                    self.damage_current_vehicle(92)
                elif target_kind == "friendly_apc":
                    self.damage_current_vehicle(82)
                elif target_kind == "pillbox":
                    self.damage_pillbox(48)
                else:
                    self.damage_player(34)
                return Task.done
            return Task.cont

        self.taskMgr.add(shell_task, f"enemy-tank-shell-{self.time}-{random.random()}")

    # ----------------------------
    # Damage
    # ----------------------------

    def damage_current_vehicle(self, amount):
        """Damage the occupied vehicle. If it is destroyed, the player dies with it."""
        if not self.in_vehicle:
            return False
        if self.current_vehicle_kind == "tank":
            self.vehicle_health = max(0, self.vehicle_health - amount)
            self.set_status(f"TANK HIT: {int(self.vehicle_health)}/{self.vehicle_max_health}", 1.2)
            destroyed = self.vehicle_health <= 0
        else:
            self.apc_health = max(0, self.apc_health - amount)
            self.set_status(f"APC HIT: {int(self.apc_health)}/{self.apc_max_health}", 1.2)
            destroyed = self.apc_health <= 0
        if destroyed:
            self.health = 0
            self.game_over = True
            self.hud_warning.setText("GAME OVER - VEHICLE DESTROYED")
            self.create_impact_effect(self.player_pos, self.enemy_tank_shell_weapon)
        return True

    def damage_player(self, amount):
        if self.game_over:
            return
        if self.in_vehicle:
            self.damage_current_vehicle(amount)
            return
        if self.pillbox_health > 0:
            return

        self.health -= amount
        self.damage_flash_timer = 0.18

        if self.health <= 0:
            self.health = 0
            self.game_over = True
            self.hud_warning.setText("GAME OVER - PLAYER DOWN")

    def damage_pillbox(self, amount):
        if self.game_over:
            return

        self.pillbox_health -= amount
        self.damage_flash_timer = 0.18

        if self.pillbox_health <= 0:
            self.pillbox_health = 0

    # ----------------------------
    # Main update
    # ----------------------------

    def update(self, task):
        dt = globalClock.getDt()
        dt = min(dt, 0.033)

        if self.paused:
            self.draw_crosshair(moving=False, firing=False)
            self.update_hud()
            self.draw_minimap()
            return Task.cont

        self.time += dt

        if self.wave_break:
            self.wave_break_timer = max(0.0, self.wave_break_timer - dt)
            if self.wave_break_timer <= 0.0:
                self.begin_next_wave()

        if not self.game_over:
            self.update_mortar_targeting(dt)
            self.update_player(dt)
            self.fire_timer = max(0.0, self.fire_timer - dt)
            self.vehicle_fire_timer = max(0.0, self.vehicle_fire_timer - dt)
            self.apc_lmg_fire_timer = max(0.0, self.apc_lmg_fire_timer - dt)
            self.grenade_cooldown = max(0.0, self.grenade_cooldown - dt)
            if self.apc_lmg_reloading:
                self.apc_lmg_reload_timer -= dt
                if self.apc_lmg_reload_timer <= 0:
                    self.finish_apc_reload()
            if self.in_vehicle and self.current_vehicle_kind == "apc" and self.keys["space"]:
                self.fire_apc_lmg()

            if self.reloading:
                self.reload_timer -= dt
                if self.reload_timer <= 0:
                    self.finish_reload()

            if self.mouse_down and not self.mortar_targeting:
                self.try_fire()

            self.update_bullets(dt)
            self.update_enemies(dt)

        self.update_explosions(dt)

        self.hit_marker_timer = max(0.0, self.hit_marker_timer - dt)
        self.damage_flash_timer = max(0.0, self.damage_flash_timer - dt)
        self.status_timer = max(0.0, self.status_timer - dt)
        self.mortar_cooldown = max(0.0, self.mortar_cooldown - dt)

        moving = any([self.keys["w"], self.keys["a"], self.keys["s"], self.keys["d"]])
        self.draw_crosshair(moving=moving, firing=self.fire_timer > 0.03)
        self.update_hud()
        self.draw_minimap()
        return Task.cont


if __name__ == "__main__":
    game = FPSGame()
    game.run()
