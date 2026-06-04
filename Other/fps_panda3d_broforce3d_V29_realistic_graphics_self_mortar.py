# BUILD_TAG: V29_REALISM_GRAPHICS_SELF_MORTAR
# BUILD_TAG: V27_SHIFT_SCOPE_STRICT_LADDERS_OBSTACLE_MAP_WT_MORTAR
# BUILD_TAG: V26_DIRECTIONAL_INCOMING_FIRE_INDICATORS
# BUILD_TAG: V25_ROOF_LOS_BLOCKERS
# fps_panda3d_broforce3d_V24_stability_collision_clipfix.py\n# V24_STABILITY_PASS_ENEMY_WALLS_CAMERA_CLIP_FIX
# Single-file Panda3D FPS / 3D Broforce-inspired level game.
# No panda3d.bullet dependency: this reverts to the previous lightweight custom physics setup.
#
# Install:
#   python3 -m pip install panda3d
#
# Run:
#   python3 fps_panda3d_broforce3d_V15_no_bullet_level_mode.py
#
# Controls:
#   WASD       camera-relative move on foot
#   Mouse      look / aim
#   Left click fire current weapon
#   1/2/3/4    Rifle / SMG / Shotgun / RPG
#   R          reload
#   G          hold/release to throw 4s fuse grenade
#   Shift      scope / circular limited-view optic
#   E          interact / enter or exit APC/tank / rescue hostage
#   X          buy/aim/confirm mortar strike ($150); arrows move target; C cancels
#   F          use medkit/ammo crate nearby
#   Space      jump on foot; APC LMG when driving APC
#   Z          switch tank ammo HE/AP while in tank
#   Shift      sprint
#   Esc        pause menu
#
# Pause menu:
#   C continue | R restart level | Q quit | M release/lock mouse

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import (
    Vec3, Vec4, Point3,
    WindowProperties,
    AmbientLight, DirectionalLight, PointLight,
    Geom, GeomNode, GeomTriangles,
    GeomVertexFormat, GeomVertexData,
    GeomVertexWriter,
    LineSegs, TextNode,
    Material, TransparencyAttrib,
    NodePath,
)

import math
import random
import hashlib
from dataclasses import dataclass, field

BUILD_MARKER = "NO_BULLET_BROFORCE3D_V20_ALL_LEVELS_CLOSE_COMBAT_MAZES_2026_05_31"


# ============================================================
# Basic geometry helpers
# ============================================================

def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def make_material(diffuse, specular=(0.08, 0.08, 0.08, 1), shininess=10):
    mat = Material()
    mat.setDiffuse(diffuse)
    mat.setAmbient((diffuse[0] * 0.45, diffuse[1] * 0.45, diffuse[2] * 0.45, 1))
    mat.setSpecular(specular)
    mat.setShininess(shininess)
    return mat


def make_box(parent, name, size=(1, 1, 1), color=(1, 1, 1, 1), pos=(0, 0, 0), hpr=(0, 0, 0), material=None):
    sx, sy, sz = size[0] / 2, size[1] / 2, size[2] / 2
    verts = [
        (-sx, -sy, -sz), ( sx, -sy, -sz), ( sx,  sy, -sz), (-sx,  sy, -sz),
        (-sx, -sy,  sz), ( sx, -sy,  sz), ( sx,  sy,  sz), (-sx,  sy,  sz),
    ]
    faces = [
        (0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
        (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0),
    ]
    fmt = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    colour = GeomVertexWriter(vdata, "color")

    face_normals = [
        (0, 0, -1), (0, 0, 1), (0, -1, 0),
        (1, 0, 0), (0, 1, 0), (-1, 0, 0)
    ]
    # Flat-shaded by duplicating vertices per face.
    index = 0
    tris = GeomTriangles(Geom.UHStatic)
    for face, n in zip(faces, face_normals):
        base = index
        for vi in face:
            vertex.addData3(*verts[vi])
            normal.addData3(*n)
            colour.addData4(*color)
            index += 1
        tris.addVertices(base, base + 1, base + 2)
        tris.addVertices(base, base + 2, base + 3)

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    np = parent.attachNewNode(node)
    np.setPos(*pos)
    np.setHpr(*hpr)
    if material:
        np.setMaterial(material, 1)
    return np


def make_sphere(game, parent, name, pos, scale, color, material=None):
    model = game.loader.loadModel("models/misc/sphere")
    model.reparentTo(parent)
    model.setName(name)
    model.setPos(pos)
    if isinstance(scale, tuple):
        model.setScale(*scale)
    else:
        model.setScale(scale)
    model.setColor(*color)
    if material:
        model.setMaterial(material, 1)
    return model


def make_cylinder_approx(parent, name, radius=1.0, depth=1.0, color=(1, 1, 1, 1), pos=(0, 0, 0), hpr=(0, 0, 0), segments=18):
    # Axis along Y. Good for barrels, pipes, logs.
    fmt = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    colour = GeomVertexWriter(vdata, "color")
    tris = GeomTriangles(Geom.UHStatic)

    # side vertices duplicated as pairs.
    for i in range(segments):
        a = math.tau * i / segments
        x = math.cos(a) * radius
        z = math.sin(a) * radius
        n = Vec3(x, 0, z)
        n.normalize()
        vertex.addData3(x, -depth / 2, z)
        normal.addData3(n)
        colour.addData4(*color)
        vertex.addData3(x, depth / 2, z)
        normal.addData3(n)
        colour.addData4(*color)

    for i in range(segments):
        a = 2 * i
        b = 2 * ((i + 1) % segments)
        tris.addVertices(a, b, a + 1)
        tris.addVertices(b, b + 1, a + 1)

    # end caps
    center_front = segments * 2
    vertex.addData3(0, -depth / 2, 0)
    normal.addData3(0, -1, 0)
    colour.addData4(*color)
    center_back = center_front + 1
    vertex.addData3(0, depth / 2, 0)
    normal.addData3(0, 1, 0)
    colour.addData4(*color)

    for i in range(segments):
        a = 2 * i
        b = 2 * ((i + 1) % segments)
        tris.addVertices(center_front, a, b)
        tris.addVertices(center_back, b + 1, a + 1)

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    np = parent.attachNewNode(node)
    np.setPos(*pos)
    np.setHpr(*hpr)
    return np


def draw_line(parent, name, points, color=(1, 1, 1, 1), thickness=2):
    if len(points) < 2:
        return parent.attachNewNode(name)
    seg = LineSegs(name)
    seg.setThickness(thickness)
    seg.setColor(*color)
    seg.moveTo(points[0][0], 0, points[0][1])
    for p in points[1:]:
        seg.drawTo(p[0], 0, p[1])
    return parent.attachNewNode(seg.create())


# ============================================================
# Data
# ============================================================

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
    knockback: float = 0.0
    projectile_scale: float = 0.045
    color: tuple = (1.0, 0.82, 0.22, 1)
    overpenetrate_mobs: bool = False


@dataclass
class LevelSpec:
    name: str
    briefing: str
    length: float
    width: float
    objective_text: str
    required_rescues: int
    required_destroy: int
    enemies: list = field(default_factory=list)
    tanks: list = field(default_factory=list)
    pickups: list = field(default_factory=list)
    obstacles: list = field(default_factory=list)
    hostages: list = field(default_factory=list)
    explosive_barrels: list = field(default_factory=list)
    apc_pos: tuple = None
    friendly_tank_pos: tuple = None
    spawners: list = field(default_factory=list)


class BulletProjectile:
    def __init__(self, pos, vel, weapon, node, owner="player", tank_shell=False, ammo_type="HE"):
        self.pos = Point3(pos)
        self.vel = Vec3(vel)
        self.weapon = weapon
        self.node = node
        self.owner = owner
        self.tank_shell = tank_shell
        self.ammo_type = ammo_type
        self.ttl = 5.0
        self.hit_ids = set()


class Grenade:
    def __init__(self, pos, vel, node, fuse_time=4.0):
        self.pos = Point3(pos)
        self.vel = Vec3(vel)
        self.node = node
        # Proper timed fuse: the grenade explodes 4 seconds after release,
        # not on impact. It can bounce while the fuse burns.
        self.timer = fuse_time
        self.bounces = 0


class Pickup:
    def __init__(self, game, kind, pos):
        self.game = game
        self.kind = kind
        self.pos = Point3(pos)
        self.taken = False
        self.node = game.render.attachNewNode(f"pickup-{kind}")
        self.node.setPos(self.pos)
        color = (0.2, 1.0, 0.25, 1) if kind == "health" else (1.0, 0.82, 0.16, 1)
        make_box(self.node, "crate", (0.65, 0.65, 0.35), color, (0, 0, 0.18))
        make_box(self.node, "mark", (0.12, 0.72, 0.08), (1, 1, 1, 1), (0, 0, 0.42))
        make_box(self.node, "mark2", (0.72, 0.12, 0.08), (1, 1, 1, 1), (0, 0, 0.42))


class Hostage:
    def __init__(self, game, pos):
        self.game = game
        self.pos = Point3(pos)
        self.rescued = False
        self.node = game.render.attachNewNode("hostage")
        self.node.setPos(self.pos)
        make_box(self.node, "body", (0.38, 0.26, 0.85), (0.1, 0.35, 0.85, 1), (0, 0, 0.55))
        make_sphere(game, self.node, "head", (0, 0, 1.12), 0.20, (0.72, 0.52, 0.34, 1))
        label = TextNode("rescue_label")
        label.setText("RESCUE")
        label.setAlign(TextNode.ACenter)
        label.setTextColor(0.2, 0.8, 1.0, 1)
        tag = self.node.attachNewNode(label)
        tag.setScale(0.32)
        tag.setPos(0, 0, 1.65)


class Destructible:
    def __init__(self, game, kind, pos, health=120):
        self.game = game
        self.kind = kind
        self.pos = Point3(pos)
        self.health = health
        self.max_health = health
        self.destroyed = False
        self.radius = 1.2
        self.node = game.render.attachNewNode(f"destructible-{kind}")
        self.node.setPos(self.pos)
        if kind == "radio":
            make_box(self.node, "radio-base", (1.4, 1.1, 1.2), (0.18, 0.2, 0.22, 1), (0, 0, 0.65))
            make_cylinder_approx(self.node, "antenna", 0.04, 2.2, (0.03, 0.03, 0.03, 1), (0, 0, 2.0), (0, 0, 0), 10)
        else:
            make_box(self.node, "generator", (1.5, 1.2, 0.9), (0.2, 0.18, 0.12, 1), (0, 0, 0.45))
            make_cylinder_approx(self.node, "pipe", 0.08, 1.8, (0.08, 0.08, 0.08, 1), (0.45, 0, 1.05), (90, 0, 0), 10)

    def damage(self, amount, hit_pos=None):
        if self.destroyed:
            return
        self.health -= amount
        self.node.setColorScale(1.5, 0.65, 0.4, 1)
        if self.health <= 0:
            self.destroyed = True
            self.node.removeNode()
            self.game.destroyed_objectives += 1
            self.game.spawn_explosion(self.pos + Vec3(0, 0, 0.9), radius=5.0, damage=75, source="objective")


class MobSpawner:
    """Visible destructible enemy spawner: activates when the player gets close."""
    next_id = 0

    def __init__(self, game, pos, kinds=("runner", "rifleman"), total=6, interval=3.0, activation_radius=36.0):
        MobSpawner.next_id += 1
        self.uid = MobSpawner.next_id
        self.game = game
        self.pos = Point3(pos)
        self.kinds = list(kinds)
        self.total = int(total)
        self.spawned = 0
        self.interval = interval
        self.activation_radius = activation_radius
        self.cooldown = random.uniform(0.4, 1.4)
        self.health = 150 + 18 * game.level_index
        self.max_health = self.health
        self.alive = True
        self.radius = 1.35
        self.node = game.render.attachNewNode(f"mob-spawner-{self.uid}")
        self.node.setPos(self.pos)
        # Small bunker/doorway-like spawn point, Minecraft-spawner inspired but military styled.
        make_box(self.node, "spawner-base", (2.2, 2.2, 0.35), (0.11, 0.11, 0.11, 1), (0, 0, 0.18))
        make_box(self.node, "spawner-core", (1.5, 1.5, 1.35), (0.23, 0.05, 0.05, 1), (0, 0, 0.95))
        make_box(self.node, "spawner-door", (1.0, 0.10, 0.9), (0.02, 0.02, 0.025, 1), (0, -0.80, 0.75))
        make_sphere(game, self.node, "red-light", (0, -0.88, 1.65), 0.16, (1.0, 0.05, 0.02, 1))
        self.bar_root = self.node.attachNewNode("spawner-hp")
        self.bar_root.setPos(0, -1.15, 2.0)
        make_box(self.bar_root, "back", (1.7, 0.06, 0.09), (0, 0, 0, 1), (0, 0, 0))
        self.bar = make_box(self.bar_root, "hp", (1.62, 0.08, 0.11), (1.0, 0.08, 0.04, 1), (0, -0.02, 0.01))

    def damage(self, amount):
        if not self.alive:
            return
        self.health -= amount
        self.node.setColorScale(1.4, 0.5, 0.5, 1)
        if self.health <= 0:
            self.alive = False
            self.node.removeNode()
            self.game.money += 90
            self.game.show_message("Spawner destroyed +$90")
            self.game.spawn_explosion(self.pos + Vec3(0, 0, 0.7), radius=4.2, damage=55, source="spawner")

    def update(self, dt):
        if not self.alive:
            return
        self.node.setColorScale(1, 1, 1, 1)
        ratio = clamp(self.health / self.max_health, 0, 1)
        self.bar.setScale(ratio, 1, 1)
        self.bar.setX(-(1 - ratio) * 0.80)
        target = self.game.get_combat_target_pos()
        dist = (Vec3(target.x - self.pos.x, target.y - self.pos.y, 0)).length()
        if dist > self.activation_radius or self.spawned >= self.total:
            return
        # Keep a modest number alive per spawner so it does not flood instantly.
        nearby_alive = 0
        for enemy in self.game.enemies:
            if enemy.alive and (enemy.pos - self.pos).length() < 26:
                nearby_alive += 1
        if nearby_alive >= 5:
            return
        self.cooldown -= dt
        if self.cooldown <= 0:
            kind = random.choice(self.kinds)
            # Spawn just outside the dark doorway.
            offset = Vec3(random.uniform(-1.1, 1.1), random.uniform(-2.2, -1.4), 0)
            sx = self.pos.x + offset.x
            sy = self.pos.y + offset.y
            if self.game.is_obstacle_blocked(sx, sy, self.game.terrain_height(sx, sy), radius=0.75, height=1.9):
                self.cooldown = 0.35
                return
            spawn_pos = Point3(sx, sy, self.game.terrain_height(sx, sy))
            mob = Enemy(self.game, spawn_pos, kind)
            mob.alert = True
            self.game.enemies.append(mob)
            self.spawned += 1
            self.cooldown = self.interval * random.uniform(0.75, 1.25)


class Enemy:
    next_id = 0

    def __init__(self, game, pos, kind="rifleman"):
        Enemy.next_id += 1
        self.uid = Enemy.next_id
        self.game = game
        self.kind = kind
        self.pos = Point3(pos)
        self.vel = Vec3(0, 0, 0)
        self.knock_time = 0.0
        self.attack_cd = random.uniform(0.4, 1.4)
        self.alive = True
        self.alert = False
        self.radius = 0.42
        self.height = 1.85
        self.reward = 18

        if kind == "runner":
            self.health = 55
            self.speed = 2.8
            self.damage = 6
            body_color = (0.52, 0.12, 0.10, 1)
        elif kind == "gunner":
            self.health = 75
            self.speed = 1.8
            self.damage = 9
            body_color = (0.18, 0.28, 0.38, 1)
            self.reward = 24
        elif kind == "bruiser":
            self.health = 130
            self.speed = 1.35
            self.damage = 12
            self.radius = 0.55
            self.height = 2.15
            body_color = (0.28, 0.18, 0.12, 1)
            self.reward = 35
        else:
            self.health = 70
            self.speed = 2.0
            self.damage = 8
            body_color = (0.35, 0.12, 0.10, 1)

        self.max_health = self.health
        self.node = game.render.attachNewNode(f"enemy-{self.uid}")
        self.node.setPos(self.pos)

        s = self.height / 1.85
        make_box(self.node, "legs", (0.42 * s, 0.30 * s, 0.72 * s), (0.12, 0.14, 0.12, 1), (0, 0, 0.36 * s))
        make_box(self.node, "torso", (0.62 * s, 0.38 * s, 0.70 * s), body_color, (0, 0, 1.03 * s))
        make_sphere(game, self.node, "head", (0, 0, 1.55 * s), 0.22 * s, (0.62, 0.43, 0.30, 1))
        make_box(self.node, "weapon", (0.12 * s, 0.75 * s, 0.10 * s), (0.04, 0.04, 0.04, 1), (0.35 * s, -0.25 * s, 1.13 * s), (12, 0, 0))

        self.bar_root = self.node.attachNewNode("healthbar-root")
        self.bar_root.setPos(0, -0.65, self.height + 0.35)
        self.bar_back = make_box(self.bar_root, "hp-back", (1.15, 0.06, 0.08), (0.02, 0.02, 0.02, 1), (0, 0, 0))
        self.bar = make_box(self.bar_root, "hp", (1.10, 0.07, 0.09), (0.05, 1.0, 0.18, 1), (0, -0.02, 0.01))

    def apply_damage(self, amount, knock_dir=None, knock_strength=0.0, zone="body"):
        if not self.alive:
            return
        if zone == "head":
            amount *= 2.6
        elif zone == "chest":
            amount *= 1.45
        self.health -= amount
        self.alert = True
        self.node.setColorScale(1.6, 0.45, 0.45, 1)
        if knock_dir is not None and knock_strength > 0:
            k = Vec3(knock_dir)
            if k.lengthSquared() > 0:
                k.normalize()
                self.vel += k * knock_strength
                self.vel.z += min(6.0, knock_strength * 0.22)
                self.knock_time = max(self.knock_time, 0.35)
        if self.health <= 0:
            self.die()

    def die(self):
        self.alive = False
        self.node.removeNode()
        self.game.kills += 1
        self.game.money += self.reward
        self.game.combo_kills += 1

    def update_bar(self):
        ratio = clamp(self.health / self.max_health, 0, 1)
        self.bar.setScale(ratio, 1, 1)
        self.bar.setX(-(1 - ratio) * 0.55)
        if ratio < 0.35:
            self.bar.setColor(1, 0.1, 0.05, 1)
        elif ratio < 0.65:
            self.bar.setColor(1, 0.75, 0.05, 1)
        else:
            self.bar.setColor(0.05, 1.0, 0.18, 1)

    def update(self, dt):
        if not self.alive:
            return

        player_target = self.game.get_combat_target_pos()
        to_player = Vec3(player_target.x - self.pos.x, player_target.y - self.pos.y, 0)
        dist = to_player.length()
        if dist < 26 or self.alert:
            self.alert = True

        # knockback/gravity pass
        if self.knock_time > 0:
            self.knock_time -= dt
            self.vel.z -= 9.81 * dt
            old_pos = Point3(self.pos)
            wanted = Point3(self.pos + self.vel * dt)
            wanted.z = max(wanted.z, self.game.terrain_height(wanted.x, wanted.y))
            self.pos = self.game.resolve_actor_move(old_pos, wanted, radius=self.radius + 0.18, height=self.height)
            if self.pos.x == old_pos.x and self.pos.y == old_pos.y:
                self.vel.x *= -0.25
                self.vel.y *= -0.25
            ground = self.game.terrain_height(self.pos.x, self.pos.y)
            if self.pos.z < ground:
                self.pos.z = ground
                self.vel.z = 0
                self.vel.x *= 0.65
                self.vel.y *= 0.65
            self.node.setPos(self.pos)
            self.update_bar()
            return

        if self.alert and dist > 0.05:
            old_pos = Point3(self.pos)
            to_player.normalize()
            movement = Vec3(0, 0, 0)
            if self.kind in ("rifleman", "gunner") and dist < 35:
                # ranged enemies stop and shoot from distance.
                desired = 7.0 if self.kind == "gunner" else 9.0
                if dist > desired:
                    movement += to_player * self.speed
                elif dist < desired - 2.0:
                    movement -= to_player * self.speed * 0.65
            else:
                movement += to_player * self.speed

            # separation from other enemies so they don't clump.
            sep = Vec3(0, 0, 0)
            for other in self.game.enemies:
                if other is self or not other.alive:
                    continue
                delta = Vec3(self.pos.x - other.pos.x, self.pos.y - other.pos.y, 0)
                d = delta.length()
                if 0.001 < d < 2.2:
                    delta.normalize()
                    sep += delta * (2.2 - d) * 0.95
            movement += sep

            wanted = Point3(self.pos.x + movement.x * dt, self.pos.y + movement.y * dt, self.pos.z)
            wanted.z = self.game.terrain_height(wanted.x, wanted.y)
            self.pos = self.game.resolve_actor_move(old_pos, wanted, radius=self.radius + 0.22, height=self.height)
            self.pos.z = self.game.terrain_height(self.pos.x, self.pos.y)
            heading = math.degrees(math.atan2(to_player.x, to_player.y))
            self.node.setH(heading)

        self.attack_cd = max(0.0, self.attack_cd - dt)
        if self.alert and self.attack_cd <= 0.0:
            if self.kind in ("rifleman", "gunner") and dist < 38:
                muzzle = self.pos + Vec3(0, 0, 1.35)
                target = self.game.get_combat_target_pos()
                if self.game.has_line_of_sight(muzzle, target):
                    self.game.enemy_fire(self, damage=8 if self.kind == "rifleman" else 13)
                    self.attack_cd = 1.15 if self.kind == "rifleman" else 0.62
                else:
                    # Can't shoot through the roof/wall. Move/reposition soon rather than firing unfairly.
                    self.attack_cd = 0.28
            elif dist < 1.7:
                self.game.damage_player(self.damage, self.pos)
                self.attack_cd = 0.75

        self.node.setPos(self.pos)
        self.node.setColorScale(1, 1, 1, 1)
        self.update_bar()


class EnemyTank:
    next_id = 0

    def __init__(self, game, pos, heading=180):
        EnemyTank.next_id += 1
        self.uid = EnemyTank.next_id
        self.game = game
        self.pos = Point3(pos)
        self.heading = heading
        self.turret_heading = heading
        self.health = 480
        self.max_health = 480
        self.alive = True
        self.alert = False
        self.fire_cd = random.uniform(2.5, 4.5)
        self.track_damage_timer = 0.0
        self.size = Vec3(2.5, 3.8, 1.55)
        self.node = game.render.attachNewNode(f"enemy-tank-{self.uid}")
        self.node.setPos(self.pos)
        self.node.setH(self.heading)
        self.create_model()
        self.bar_root = self.node.attachNewNode("tank-hp-root")
        self.bar_root.setPos(0, -2.8, 2.4)
        make_box(self.bar_root, "back", (2.3, 0.08, 0.12), (0, 0, 0, 1), (0, 0, 0))
        self.bar = make_box(self.bar_root, "hp", (2.2, 0.10, 0.14), (1.0, 0.2, 0.08, 1), (0, -0.02, 0.02))

    def create_model(self):
        dark = (0.14, 0.16, 0.13, 1)
        green = (0.22, 0.29, 0.18, 1)
        tan = (0.30, 0.32, 0.24, 1)
        make_box(self.node, "lower-hull", (2.8, 4.2, 0.75), dark, (0, 0, 0.55))
        make_box(self.node, "upper-hull", (2.35, 3.55, 0.75), green, (0, 0.05, 1.10))
        make_box(self.node, "front-glacis", (2.25, 0.55, 0.65), tan, (0, -1.85, 1.35), (0, 18, 0))
        make_box(self.node, "turret", (1.65, 1.55, 0.70), green, (0, -0.25, 1.78))
        make_box(self.node, "mantlet", (0.85, 0.25, 0.42), dark, (0, -1.12, 1.78))
        self.barrel = make_cylinder_approx(self.node, "main-gun", 0.10, 3.2, (0.06, 0.06, 0.05, 1), (0, -2.55, 1.78), (90, 0, 0), 18)
        make_box(self.node, "muzzle-brake", (0.40, 0.16, 0.22), (0.04, 0.04, 0.04, 1), (0, -4.12, 1.78))
        for side in [-1, 1]:
            make_box(self.node, "track", (0.45, 4.45, 0.65), (0.06, 0.065, 0.06, 1), (side * 1.45, 0, 0.47))
            for i in range(5):
                make_sphere(self.game, self.node, "road-wheel", (side * 1.47, -1.5 + i * 0.75, 0.40), (0.20, 0.06, 0.20), (0.02, 0.02, 0.02, 1))
        for i in range(6):
            make_box(self.node, "era", (0.34, 0.12, 0.20), (0.28, 0.34, 0.22, 1), (-0.9 + i * 0.36, -1.22, 1.62))
        make_box(self.node, "optic", (0.18, 0.18, 0.12), (0.02, 0.02, 0.025, 1), (0.45, -0.9, 2.20))
        make_cylinder_approx(self.node, "antenna", 0.015, 2.2, (0.02, 0.02, 0.02, 1), (-0.65, 0.45, 2.45), (0, 0, 0), 8)

    def damage(self, amount, weapon_name="", hit_pos=None, ammo_type="HE"):
        if not self.alive:
            return

        zone = "hull"
        mult = 1.0
        if hit_pos is not None:
            rel = hit_pos - self.pos
            local_y = rel.y * math.cos(math.radians(self.heading)) + rel.x * math.sin(math.radians(self.heading))
            local_x = rel.x * math.cos(math.radians(self.heading)) - rel.y * math.sin(math.radians(self.heading))
            if abs(local_x) > 1.15 and rel.z < 0.95:
                zone = "track"
                mult = 0.35
                self.track_damage_timer = 6.0
            elif rel.z > 1.45:
                zone = "turret"
                mult = 1.85
            elif local_y > 1.15:
                zone = "engine/rear"
                mult = 1.75
            elif local_y < -1.25:
                zone = "front armour"
                mult = 0.45

        # Small arms/APC LMG do negligible damage to tanks.
        if weapon_name in ("Rifle", "SMG", "Shotgun", "APC LMG"):
            amount = min(amount * 0.04, 4)
        elif weapon_name == "RPG":
            amount *= 1.35
        elif ammo_type == "AP":
            amount *= 1.55
        elif ammo_type == "HE":
            amount *= 0.85

        amount *= mult
        self.health -= amount
        self.alert = True
        self.fire_cd = min(self.fire_cd, random.uniform(1.0, 1.8))
        self.game.show_message(f"Tank hit: {zone}  -{int(amount)}")
        if self.health <= 0:
            self.die()

    def die(self):
        self.alive = False
        self.node.removeNode()
        self.game.kills += 1
        self.game.money += 150
        self.game.spawn_explosion(self.pos + Vec3(0, 0, 1.0), radius=7.0, damage=120, source="tank")

    def update(self, dt):
        if not self.alive:
            return
        self.track_damage_timer = max(0.0, self.track_damage_timer - dt)
        target = self.game.get_combat_target_pos()
        to_target = Vec3(target.x - self.pos.x, target.y - self.pos.y, 0)
        dist = to_target.length()
        if dist < 36:
            self.alert = True
        if self.alert and dist > 0.01:
            to_target.normalize()
            self.turret_heading = math.degrees(math.atan2(to_target.x, to_target.y))
            # slow tank crawl toward player/extraction path
            speed = 0.42 if self.track_damage_timer <= 0 else 0.12
            if dist > 18:
                self.pos += to_target * speed * dt
                self.pos.z = self.game.terrain_height(self.pos.x, self.pos.y)
                self.node.setPos(self.pos)
                self.heading = self.turret_heading
                self.node.setH(self.heading)

            self.fire_cd -= dt
            if self.fire_cd <= 0 and dist < 42:
                self.game.enemy_tank_fire(self)
                self.fire_cd = random.uniform(3.0, 5.0)

        ratio = clamp(self.health / self.max_health, 0, 1)
        self.bar.setScale(ratio, 1, 1)
        self.bar.setX(-(1 - ratio) * 1.1)


class FriendlyTank:
    def __init__(self, game, pos):
        self.game = game
        self.pos = Point3(pos)
        self.heading = 0.0
        self.turret_heading = 0.0
        self.health = 950
        self.max_health = 950
        self.fire_cd = 0.0
        self.reload_timer = 0.0
        self.ammo_type = "HE"
        self.ammo_types = ["HE", "AP"]
        self.occupied = False
        self.node = game.render.attachNewNode("friendly-tank")
        self.node.setPos(self.pos)
        self.create_model()

    def create_model(self):
        olive = (0.20, 0.31, 0.17, 1)
        dark = (0.055, 0.060, 0.052, 1)
        tan = (0.29, 0.36, 0.22, 1)
        make_box(self.node, "lower-hull", (3.0, 4.6, 0.75), dark, (0, 0, 0.52))
        make_box(self.node, "upper-hull", (2.55, 3.65, 0.76), olive, (0, 0.05, 1.12))
        make_box(self.node, "sloped-glacis", (2.35, 0.65, 0.55), tan, (0, -1.75, 1.32), (0, 18, 0))
        make_box(self.node, "engine-deck", (2.2, 1.0, 0.18), (0.10, 0.12, 0.09, 1), (0, 1.35, 1.55))
        self.turret = self.node.attachNewNode("friendly-turret")
        self.turret.setPos(0, -0.20, 1.78)
        make_box(self.turret, "turret-body", (1.75, 1.65, 0.72), olive, (0, 0, 0))
        make_box(self.turret, "mantlet", (0.85, 0.24, 0.40), dark, (0, -0.92, 0.02))
        make_cylinder_approx(self.turret, "cannon", 0.105, 3.55, (0.05, 0.052, 0.045, 1), (0, -2.55, 0.02), (90, 0, 0), 20)
        make_box(self.turret, "muzzle-brake", (0.48, 0.18, 0.24), dark, (0, -4.30, 0.02))
        make_box(self.turret, "optic", (0.18, 0.20, 0.13), (0.02, 0.03, 0.035, 1), (0.52, -0.70, 0.44))
        make_cylinder_approx(self.turret, "antenna", 0.014, 2.1, dark, (-0.68, 0.55, 0.82), (0, 0, 0), 8)
        for side in [-1, 1]:
            make_box(self.node, "track", (0.50, 4.85, 0.70), dark, (side * 1.55, 0, 0.48))
            make_box(self.node, "side-skirt", (0.16, 4.35, 0.46), tan, (side * 1.36, 0.05, 0.82))
            for i in range(6):
                make_sphere(self.game, self.node, "road-wheel", (side * 1.58, -1.75 + i * 0.70, 0.42), (0.20, 0.055, 0.20), (0.018, 0.018, 0.018, 1))
        for i in range(7):
            make_box(self.node, "era-block", (0.31, 0.12, 0.20), (0.28, 0.36, 0.22, 1), (-0.95 + i * 0.32, -1.23, 1.65))
        make_box(self.node, "blue-friendly-panel", (0.55, 0.12, 0.16), (0.1, 0.55, 1.0, 1), (0, 1.90, 1.45))

    def update(self, dt):
        self.fire_cd = max(0.0, self.fire_cd - dt)
        self.reload_timer = max(0.0, self.reload_timer - dt)
        if self.occupied:
            # Hull driving uses the hull heading. Turret/cannon aims where the mouse camera points.
            drive = 0.0
            if self.game.keys["w"]:
                drive += 1.0
            if self.game.keys["s"]:
                drive -= 1.0
            if self.game.keys["a"]:
                self.heading += 58 * dt
            if self.game.keys["d"]:
                self.heading -= 58 * dt
            forward = Vec3(-math.sin(math.radians(self.heading)), math.cos(math.radians(self.heading)), 0)
            self.pos += forward * drive * 5.2 * dt
            self.pos.x = clamp(self.pos.x, -self.game.level.width / 2 + 2.2, self.game.level.width / 2 - 2.2)
            self.pos.y = clamp(self.pos.y, -self.game.level.length / 2 + 2.0, self.game.level.length / 2 - 2.0)
            self.pos.z = self.game.terrain_height(self.pos.x, self.pos.y)
            self.turret_heading = -self.game.yaw
            self.node.setPos(self.pos)
            self.node.setH(self.heading)
            try:
                self.turret.setH(self.turret_heading - self.heading)
            except Exception:
                pass
            self.game.player_pos = Point3(self.pos.x, self.pos.y, self.pos.z + 2.25)
            self.game.camera.setPos(self.game.player_pos)
            self.game.camera.setHpr(self.game.yaw, self.game.pitch, 0)
        else:
            self.node.setPos(self.pos)
            self.node.setH(self.heading)

    def switch_ammo(self):
        if self.reload_timer > 0:
            return
        self.ammo_type = "AP" if self.ammo_type == "HE" else "HE"
        self.game.show_message(f"Tank ammo: {self.ammo_type}")

    def fire_cannon(self):
        if not self.occupied or self.fire_cd > 0 or self.reload_timer > 0:
            return
        self.fire_cd = 0.05
        self.reload_timer = 1.30
        direction = self.game.camera.getQuat(self.game.render).getForward()
        direction.normalize()
        start = self.pos + Vec3(0, 0, 1.85) + direction * 2.4
        if self.ammo_type == "AP":
            shell = Weapon(
                "Tank AP", 365, 225, 45, 1, 0, 0, 0.035, 0.055,
                splash_radius=0.0, splash_damage=0, knockback=1.2,
                projectile_scale=0.075, color=(0.84, 0.84, 0.70, 1), overpenetrate_mobs=True,
            )
        else:
            shell = Weapon(
                "Tank HE", 210, 108, 45, 1, 0, 0, 0.10, 0.10,
                splash_radius=6.5, splash_damage=155, knockback=8.5,
                projectile_scale=0.115, color=(1.0, 0.34, 0.08, 1),
            )
        self.game.spawn_projectile(start, direction, shell, owner="player", tank_shell=True, ammo_type=self.ammo_type)


class FriendlyAPC:
    def __init__(self, game, pos):
        self.game = game
        self.pos = Point3(pos)
        self.heading = 0.0
        self.turret_heading = 0.0
        self.health = 650
        self.max_health = 650
        self.ammo = 200
        self.reserve = 600
        self.mag_size = 200
        self.fire_cd = 0.0
        self.reload_timer = 0.0
        self.occupied = False
        self.node = game.render.attachNewNode("friendly-apc")
        self.node.setPos(self.pos)
        self.create_model()

    def create_model(self):
        olive = (0.18, 0.28, 0.16, 1)
        dark = (0.06, 0.07, 0.06, 1)
        make_box(self.node, "hull", (2.6, 4.2, 1.25), olive, (0, 0, 0.8))
        make_box(self.node, "sloped-front", (2.35, 0.75, 0.70), (0.20, 0.30, 0.18, 1), (0, -1.95, 1.10), (0, 16, 0))
        make_box(self.node, "turret", (1.2, 1.0, 0.45), olive, (0, -0.35, 1.65))
        make_cylinder_approx(self.node, "lmg", 0.055, 2.0, dark, (0, -1.65, 1.70), (90, 0, 0), 12)
        for side in [-1, 1]:
            make_box(self.node, "wheel-row", (0.35, 4.1, 0.55), dark, (side * 1.45, 0, 0.45))
            for i in range(4):
                make_sphere(self.game, self.node, "wheel", (side * 1.48, -1.35 + i * 0.9, 0.45), (0.28, 0.08, 0.28), (0.02, 0.02, 0.02, 1))
        make_box(self.node, "blue-marker", (0.4, 0.4, 0.08), (0.1, 0.55, 1.0, 1), (0, 0.75, 1.55))

    def update(self, dt):
        self.fire_cd = max(0.0, self.fire_cd - dt)
        if self.reload_timer > 0:
            self.reload_timer -= dt
            if self.reload_timer <= 0:
                need = self.mag_size - self.ammo
                take = min(need, self.reserve)
                self.ammo += take
                self.reserve -= take
        if self.occupied:
            yaw = math.radians(self.game.yaw)
            self.turret_heading = -self.game.yaw
            drive = 0.0
            if self.game.keys["w"]:
                drive += 1.0
            if self.game.keys["s"]:
                drive -= 1.0
            if self.game.keys["a"]:
                self.heading += 80 * dt
            if self.game.keys["d"]:
                self.heading -= 80 * dt
            forward = Vec3(-math.sin(math.radians(self.heading)), math.cos(math.radians(self.heading)), 0)
            self.pos += forward * drive * 7.2 * dt
            self.pos.z = self.game.terrain_height(self.pos.x, self.pos.y)
            self.node.setPos(self.pos)
            self.node.setH(self.heading)
            self.game.player_pos = Point3(self.pos.x, self.pos.y, self.pos.z + 2.15)
            self.game.camera.setPos(self.game.player_pos)
            self.game.camera.setHpr(self.game.yaw, self.game.pitch, 0)

    def fire_lmg(self):
        if not self.occupied or self.fire_cd > 0 or self.reload_timer > 0:
            return
        if self.ammo <= 0:
            self.reload_timer = 3.2 if self.reserve > 0 else 0
            return
        self.ammo -= 1
        self.fire_cd = 1.0 / 12.0
        direction = self.game.camera.getQuat(self.game.render).getForward()
        start = self.pos + Vec3(0, 0, 1.7) + direction * 1.3
        w = Weapon("APC LMG", 10, 115, 720, 200, 0, 3.2, 1.2, 0.25, knockback=0.5, projectile_scale=0.035, color=(1.0, 0.82, 0.18, 1))
        self.game.spawn_projectile(start, direction, w, owner="player")


# ============================================================
# Main game
# ============================================================

class FPSGame(ShowBase):
    def __init__(self):
        super().__init__()
        self.disableMouse()
        self.render.setShaderAuto()
        self.setBackgroundColor(0.46, 0.65, 0.84, 1)

        self.window_locked = True
        self.setup_window()

        self.levels = self.create_level_specs()
        self.level_index = 0
        self.level = None

        self.keys = {k: False for k in ["w", "a", "s", "d", "space", "shift"]}
        self.mouse_down = False
        self.paused = False
        self.mouse_free = False
        self.scoped = False
        self.yaw = 0.0
        self.pitch = 0.0
        self.mouse_sensitivity = 0.12
        self.normal_fov = 75
        self.scope_fov = 28

        self.player_pos = Point3(0, 0, 1.65)
        self.player_vel = Vec3(0, 0, 0)
        self.eye_height = 1.65
        self.grounded = False
        self.health = 100
        self.max_health = 100
        self.money = 0
        self.kills = 0
        self.combo_kills = 0
        self.grenades = 4
        self.destroyed_objectives = 0
        self.rescued_hostages = 0
        self.extraction_ready = False
        self.level_complete_timer = 0.0
        self.message_timer = 0.0
        self.message_text = ""

        self.weapons = [
            Weapon("Rifle", 32, 150, 430, 24, 144, 1.55, 0.16, 0.18, knockback=0.7, projectile_scale=0.035),
            Weapon("SMG", 17, 92, 760, 40, 240, 1.25, 1.20, 0.28, knockback=0.35, projectile_scale=0.032),
            Weapon("Shotgun", 13, 72, 95, 7, 49, 2.05, 4.6, 0.42, pellets=9, knockback=1.25, projectile_scale=0.030),
            Weapon("RPG", 155, 58, 28, 1, 9, 2.7, 0.18, 0.0, splash_radius=8.0, splash_damage=185, knockback=9.5, projectile_scale=0.13, color=(1.0, 0.25, 0.08, 1)),
        ]
        self.current_weapon = 0
        self.magazines = [w.mag_size for w in self.weapons]
        self.reserves = [w.reserve for w in self.weapons]
        self.fire_timer = 0.0
        self.reload_timer = 0.0
        self.reload_total = 0.0
        self.reloading = False

        self.enemies = []
        self.enemy_tanks = []
        self.projectiles = []
        self.enemy_projectiles = []
        self.grenade_objects = []
        self.pickups = []
        self.hostages = []
        self.destructibles = []
        self.obstacles = []
        self.walkable_surfaces = []
        self.ladders = []
        self.ladder_exit_cooldown = 0.0
        self.explosive_barrels = []
        self.spawners = []
        self.active_explosions = []
        self.active_dust = []
        self.active_tracers = []
        self.apc = None
        self.in_apc = False
        self.friendly_tank = None
        self.in_tank = False
        self.friendly_tank = None
        self.in_tank = False
        self.grenade_charging = False
        self.grenade_charge = 0.0
        self.mortar_cost = 150
        self.mortar_cooldown = 0.0
        self.mortar_targeting = False
        self.mortar_target = Point3(0, 0, 0)
        self.mortar_pending = []
        self.mortar_marker_np = None

        self.crosshair_np = None
        self.reload_ring_np = None
        self.minimap_np = None
        self.scope_np = None
        self.damage_indicator_np = None
        self.damage_indicators = []

        self.setup_lighting()
        self.setup_input()
        self.setup_hud()
        self.load_level(0)
        self.taskMgr.add(self.update, "update")

    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------

    def setup_window(self):
        props = WindowProperties()
        props.setCursorHidden(True)
        self.win.requestProperties(props)
        # Reduce near-plane clipping artefacts when aiming close to walls,
        # and use a long far plane for the larger maze levels.
        try:
            self.cam.node().getLens().setNearFar(0.08, 900.0)
        except Exception:
            pass
        self.center_x = self.win.getXSize() // 2
        self.center_y = self.win.getYSize() // 2
        self.win.movePointer(0, self.center_x, self.center_y)

    def setup_lighting(self):
        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.44, 0.44, 0.44, 1))
        self.render.setLight(self.render.attachNewNode(ambient))
        sun = DirectionalLight("sun")
        sun.setColor(Vec4(0.98, 0.92, 0.78, 1))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(-35, -55, 0)
        self.render.setLight(sun_np)
        fill = PointLight("blue-fill")
        fill.setColor(Vec4(0.10, 0.13, 0.18, 1))
        fill_np = self.render.attachNewNode(fill)
        fill_np.setPos(-40, -40, 25)
        self.render.setLight(fill_np)

    def setup_input(self):
        # Shift is now the optic/scope key, so it is no longer used for sprint.
        # This avoids conflicting movement and aiming behaviour.
        for key in ["w", "a", "s", "d", "space"]:
            self.accept(key, self.set_key, [key, True])
            self.accept(key + "-up", self.set_key, [key, False])
        self.accept("shift", self.toggle_scope)
        self.accept("shift-up", lambda: None)
        self.accept("mouse1", self.set_mouse, [True])
        self.accept("mouse1-up", self.set_mouse, [False])
        self.accept("1", self.switch_weapon, [0])
        self.accept("2", self.switch_weapon, [1])
        self.accept("3", self.switch_weapon, [2])
        self.accept("4", self.switch_weapon, [3])
        self.accept("r", self.reload_or_vehicle_reload)
        self.accept("g", self.start_grenade_charge)
        self.accept("g-up", self.release_grenade)
        self.accept("x", self.mortar_action)
        self.accept("enter", self.mortar_action)
        self.accept("z", self.switch_tank_ammo)
        self.accept("e", self.interact)
        self.accept("f", self.use_nearby_pickup)
        self.accept("escape", self.toggle_pause)
        self.accept("c", self.cancel_or_continue)
        self.accept("m", self.toggle_mouse_free)
        self.accept("q-up", lambda: None)

    def setup_hud(self):
        self.hud_main = OnscreenText("", pos=(-1.30, -0.70), scale=0.045, align=TextNode.ALeft, fg=(1, 1, 1, 1), mayChange=True)
        self.hud_top = OnscreenText("", pos=(-1.30, 0.90), scale=0.043, align=TextNode.ALeft, fg=(1, 1, 1, 1), mayChange=True)
        self.hud_obj = OnscreenText("", pos=(0, 0.88), scale=0.045, align=TextNode.ACenter, fg=(0.9, 0.95, 1, 1), mayChange=True)
        self.hud_message = OnscreenText("", pos=(0, 0.72), scale=0.052, align=TextNode.ACenter, fg=(1, 0.85, 0.25, 1), mayChange=True)
        self.hud_pause = OnscreenText("", pos=(0, 0.18), scale=0.065, align=TextNode.ACenter, fg=(1, 1, 1, 1), mayChange=True)

    # ------------------------------------------------------------
    # Levels
    # ------------------------------------------------------------

    def create_level_specs(self):
        # V20: all levels use close-combat maze layouts.
        # Level 1 is medium difficulty; later levels increase maze density, spawner pressure,
        # armour threats, and objective count.
        return [
            LevelSpec(
                name="1 - Maze Outpost",
                briefing="Close-combat raid. Start between your tank and APC, enter the two-floor building, destroy the radio, rescue the hostage, then push through the green extraction zone.",
                length=132,
                width=74,
                objective_text="Clear the maze outpost: rescue 1 hostage + destroy radio + reach extraction",
                required_rescues=1,
                required_destroy=1,
                enemies=[
                    ("rifleman", -9, -31), ("runner", 13, -22), ("rifleman", -18, -2),
                    ("gunner", 17, 18), ("runner", -10, 34),
                ],
                spawners=[
                    (-24, -4, ("runner", "rifleman"), 6, 3.0, 30),
                    (20, 16, ("rifleman", "gunner"), 6, 3.3, 32),
                    (-19, 43, ("runner", "rifleman"), 5, 3.0, 30),
                    (21, 53, ("gunner", "rifleman"), 4, 3.6, 28),
                ],
                pickups=[("health", -25, -18), ("ammo", 24, 8), ("ammo", -16, 48)],
                hostages=[(-16, -3)],
                obstacles=[
                    # Small cover pieces only; the main maze is built by create_level_one_maze().
                    (-6, -38, 2.8, 1.0, 1.0), (7, -29, 3.2, 1.0, 1.0),
                    (-24, 14, 3.2, 1.0, 1.0), (25, 33, 3.2, 1.0, 1.0),
                    (0, 56, 5.5, 1.0, 1.2),
                ],
                explosive_barrels=[(-23, 7), (23, 25), (-10, 53)],
                apc_pos=(18, -56),
                friendly_tank_pos=(-18, -56),
            ),
            LevelSpec(
                name="2 - Trench Road",
                briefing="Move along the trench road. Use the captured APC if you want. Destroy the generator and escape.",
                length=145,
                width=70,
                objective_text="Destroy generator + reach extraction",
                required_rescues=0,
                required_destroy=1,
                enemies=[
                    ("runner", -22, -18), ("rifleman", 20, -8), ("gunner", 4, 10),
                ],
                spawners=[
                    (-22, 22, ("runner", "bruiser"), 7, 2.7, 36),
                    (17, 45, ("rifleman", "gunner"), 7, 3.1, 38),
                    (0, 70, ("runner", "rifleman", "gunner"), 5, 2.8, 34),
                ],
                tanks=[(12, 52, 180)],
                pickups=[("ammo", -18, 20), ("health", 18, 48)],
                obstacles=[
                    (-25, -8, 8, 1.2, 1.0), (22, 4, 1.2, 8, 1.0),
                    (-10, 26, 12, 1.2, 1.1), (18, 40, 1.2, 8, 1.0),
                    (-20, 62, 8, 1.2, 1.0),
                ],
                explosive_barrels=[(-4, 18), (16, 42), (-12, 60)],
                apc_pos=(-23, -34),
                friendly_tank_pos=(13, -46),
            ),
            LevelSpec(
                name="3 - Depot Raid",
                briefing="Break through the depot, destroy both targets, defeat armour, and reach the final extraction pad.",
                length=160,
                width=76,
                objective_text="Destroy radio + generator + reach final extraction",
                required_rescues=0,
                required_destroy=2,
                enemies=[
                    ("rifleman", -20, -24), ("gunner", 22, -14), ("runner", -10, 0),
                ],
                spawners=[
                    (-25, 22, ("runner", "rifleman"), 8, 2.5, 38),
                    (24, 50, ("gunner", "rifleman"), 8, 2.9, 40),
                    (-18, 78, ("runner", "bruiser", "gunner"), 8, 2.6, 38),
                    (18, 105, ("rifleman", "gunner", "bruiser"), 7, 2.7, 40),
                ],
                tanks=[(-18, 48, 160), (20, 78, 190)],
                pickups=[("ammo", 0, 20), ("health", -20, 58), ("ammo", 20, 88)],
                obstacles=[
                    (-28, -10, 10, 1.2, 1.2), (26, 2, 1.2, 12, 1.0),
                    (-14, 28, 14, 1.2, 1.1), (16, 38, 1.2, 12, 1.0),
                    (-26, 70, 12, 1.2, 1.2), (25, 86, 1.2, 10, 1.0),
                    (0, 102, 24, 1.2, 1.2),
                ],
                explosive_barrels=[(0, 12), (12, 42), (-18, 72), (8, 92)],
                apc_pos=(-28, -36),
                friendly_tank_pos=(16, -50),
            ),
            LevelSpec(
                name="4 - Factory Maze",
                briefing="Industrial close quarters: move through factory lanes, destroy the generator room, and reach extraction.",
                length=170,
                width=80,
                objective_text="Destroy generator + survive the factory spawners + extract",
                required_rescues=0,
                required_destroy=1,
                enemies=[("rifleman", -18, -28), ("gunner", 18, -18), ("runner", 0, 10)],
                spawners=[
                    (-28, 12, ("runner", "rifleman"), 8, 2.8, 36),
                    (28, 38, ("gunner", "rifleman"), 8, 3.0, 38),
                    (-22, 72, ("runner", "bruiser"), 7, 2.7, 38),
                ],
                tanks=[(24, 66, 185)],
                pickups=[("health", -24, 32), ("ammo", 20, 58), ("ammo", -12, 88)],
                obstacles=[
                    (-26, -4, 13, 1.2, 1.2), (16, 8, 1.2, 14, 1.1),
                    (-8, 34, 18, 1.2, 1.2), (28, 52, 1.2, 16, 1.1),
                    (-24, 82, 12, 1.2, 1.2),
                ],
                explosive_barrels=[(-18, 12), (21, 42), (-7, 74), (18, 94)],
                apc_pos=(-25, -42),
                friendly_tank_pos=(18, -52),
            ),
            LevelSpec(
                name="5 - Armour Courtyard",
                briefing="A hard courtyard with armour. Use AP tank rounds and mortar calls to break through.",
                length=185,
                width=86,
                objective_text="Destroy 2 objectives + defeat armour + extract",
                required_rescues=0,
                required_destroy=2,
                enemies=[("rifleman", -26, -30), ("gunner", 24, -20), ("bruiser", 0, 5)],
                spawners=[
                    (-30, 20, ("rifleman", "runner"), 9, 2.7, 40),
                    (30, 48, ("gunner", "rifleman"), 9, 2.9, 42),
                    (-22, 88, ("bruiser", "runner", "rifleman"), 8, 2.7, 40),
                    (24, 118, ("gunner", "bruiser"), 6, 3.1, 38),
                ],
                tanks=[(-22, 52, 210), (24, 92, 220)],
                pickups=[("ammo", 0, 30), ("health", -24, 70), ("ammo", 24, 112)],
                obstacles=[
                    (-28, -5, 16, 1.2, 1.2), (26, 14, 1.2, 16, 1.0),
                    (-18, 42, 18, 1.2, 1.2), (20, 70, 1.2, 18, 1.0),
                    (-26, 102, 16, 1.2, 1.2), (0, 130, 24, 1.2, 1.2),
                ],
                explosive_barrels=[(-8, 20), (18, 58), (-20, 84), (12, 124)],
                apc_pos=(-28, -42),
                friendly_tank_pos=(18, -55),
            ),
            LevelSpec(
                name="6 - Command Labyrinth",
                briefing="Final maze: heavy spawners, multiple armour threats, and a winding route through the command compound.",
                length=205,
                width=90,
                objective_text="Destroy command radio + generator + reach final extraction",
                required_rescues=0,
                required_destroy=2,
                enemies=[("rifleman", -28, -34), ("gunner", 28, -26), ("runner", -8, -2), ("bruiser", 12, 18)],
                spawners=[
                    (-32, 18, ("runner", "rifleman"), 10, 2.5, 42),
                    (32, 46, ("gunner", "rifleman"), 10, 2.7, 44),
                    (-28, 82, ("runner", "bruiser", "rifleman"), 9, 2.5, 42),
                    (28, 120, ("gunner", "bruiser"), 9, 2.8, 44),
                    (0, 150, ("runner", "gunner", "bruiser"), 7, 2.6, 38),
                ],
                tanks=[(-25, 58, 230), (24, 103, 240), (0, 142, 250)],
                pickups=[("health", -25, 40), ("ammo", 24, 68), ("ammo", -24, 118), ("health", 18, 146)],
                obstacles=[
                    (-28, -12, 18, 1.2, 1.2), (30, 6, 1.2, 18, 1.0),
                    (-22, 40, 20, 1.2, 1.2), (22, 68, 1.2, 18, 1.0),
                    (-30, 100, 18, 1.2, 1.2), (26, 128, 1.2, 18, 1.0),
                    (0, 158, 26, 1.2, 1.2),
                ],
                explosive_barrels=[(-12, 18), (18, 44), (-18, 84), (18, 116), (0, 150)],
                apc_pos=(-30, -44),
                friendly_tank_pos=(20, -58),
            ),
        ]

    def clear_level_nodes(self):
        # Remove old level objects before resetting lists, so completed/restarted
        # levels do not leave invisible or visible stale nodes behind.
        for enemy in getattr(self, "enemies", []):
            try:
                enemy.node.removeNode()
            except Exception:
                pass
        for tank in getattr(self, "enemy_tanks", []):
            try:
                tank.node.removeNode()
            except Exception:
                pass
        for obj_list in [getattr(self, "projectiles", []), getattr(self, "enemy_projectiles", []), getattr(self, "grenade_objects", [])]:
            for obj in obj_list:
                try:
                    obj.node.removeNode()
                except Exception:
                    pass
        for obj_list in [getattr(self, "pickups", []), getattr(self, "hostages", []), getattr(self, "destructibles", []), getattr(self, "explosive_barrels", []), getattr(self, "spawners", [])]:
            for obj in obj_list:
                try:
                    obj.node.removeNode()
                except Exception:
                    pass
        for fx in getattr(self, "active_explosions", []):
            try:
                fx["root"].removeNode()
            except Exception:
                pass
        for fx in getattr(self, "active_dust", []):
            try:
                fx["root"].removeNode()
            except Exception:
                pass
        if getattr(self, "apc", None):
            try:
                self.apc.node.removeNode()
            except Exception:
                pass
        if getattr(self, "friendly_tank", None):
            try:
                self.friendly_tank.node.removeNode()
            except Exception:
                pass

        for attr in ["level_root", "terrain", "sky", "exit_node"]:
            if hasattr(self, attr):
                try:
                    getattr(self, attr).removeNode()
                except Exception:
                    pass
        self.enemies.clear()
        self.enemy_tanks.clear()
        self.projectiles.clear()
        self.enemy_projectiles.clear()
        self.grenade_objects.clear()
        self.pickups.clear()
        self.hostages.clear()
        self.destructibles.clear()
        self.obstacles.clear()
        self.walkable_surfaces.clear()
        self.ladders.clear()
        self.ladder_exit_cooldown = 0.0
        self.explosive_barrels.clear()
        self.spawners.clear()
        self.mortar_pending.clear()
        self.mortar_targeting = False
        if getattr(self, "mortar_marker_np", None):
            try:
                self.mortar_marker_np.removeNode()
            except Exception:
                pass
            self.mortar_marker_np = None
        self.active_explosions.clear()
        self.active_dust.clear()
        self.active_tracers.clear()
        self.damage_indicators.clear()
        if self.damage_indicator_np:
            try:
                self.damage_indicator_np.removeNode()
            except Exception:
                pass
            self.damage_indicator_np = None
        self.apc = None
        self.in_apc = False
        self.friendly_tank = None
        self.in_tank = False
        self.grenade_charging = False
        self.grenade_charge = 0.0
        self.grenade_charging = False
        self.grenade_charge = 0.0

    def load_level(self, index):
        self.clear_level_nodes()
        self.level_index = index
        self.level = self.levels[index]
        self.level_root = self.render.attachNewNode("level-root")
        self.destroyed_objectives = 0
        self.rescued_hostages = 0
        self.extraction_ready = False
        self.level_complete_timer = 0.0
        self.health = min(self.max_health, max(self.health, 85))
        self.grenades = max(self.grenades, 4)
        self.player_pos = Point3(0, -self.level.length / 2 + 8, self.terrain_height(0, -self.level.length / 2 + 8) + self.eye_height)
        self.player_vel = Vec3(0, 0, 0)
        self.yaw = 0
        self.pitch = 0
        self.create_sky()
        self.create_terrain()
        self.create_level_objects()
        self.camera.setPos(self.player_pos)
        self.camera.setHpr(self.yaw, self.pitch, 0)
        self.show_message(self.level.briefing, 4.5)

    def create_sky(self):
        self.sky = make_sphere(self, self.render, "sky", (0, 0, 0), 420, (0.46, 0.65, 0.84, 1))
        self.sky.setTwoSided(True)
        self.sky.setLightOff()
        self.sky.setBin("background", 0)
        self.sky.setDepthWrite(False)

    def terrain_height(self, x, y):
        if not self.level:
            return 0.0
        # Flatter than the pillbox versions, with shallow trenches and road cuts.
        h = (
            0.45 * math.sin(x * 0.09) * math.cos(y * 0.045)
            + 0.22 * math.sin((x + y) * 0.075)
            + 0.12 * math.sin(y * 0.16)
        )
        # Road/trench along the centre, slightly depressed.
        road = -0.35 * math.exp(-(x * x) / 85.0)
        # A few side berms.
        berms = 0.22 * math.exp(-((abs(x) - self.level.width * 0.38) ** 2) / 20.0)
        return h + road + berms

    def surface_ground_height(self, x, y, current_z=None):
        """Ground height including walkable stairs/roof surfaces for the player."""
        g = self.terrain_height(x, y)
        for surf in self.walkable_surfaces:
            sx = surf.get("sx", 0)
            sy = surf.get("sy", 0)
            if abs(x - surf["x"]) > sx / 2 or abs(y - surf["y"]) > sy / 2:
                continue
            kind = surf.get("kind", "flat")
            if kind == "stair":
                axis = surf.get("axis", "y")
                if axis == "y":
                    start = surf["y"] - sy / 2
                    t = clamp((y - start) / max(0.001, sy), 0.0, 1.0)
                else:
                    start = surf["x"] - sx / 2
                    t = clamp((x - start) / max(0.001, sx), 0.0, 1.0)
                hz = surf["z0"] + (surf["z1"] - surf["z0"]) * t
                g = max(g, hz)
            else:
                min_z = surf.get("min_z", -999.0)
                if current_z is None or current_z >= min_z:
                    g = max(g, surf.get("z", g))
        return g

    def add_walkable_surface(self, kind, x, y, sx, sy, **kwargs):
        data = {"kind": kind, "x": x, "y": y, "sx": sx, "sy": sy}
        data.update(kwargs)
        self.walkable_surfaces.append(data)

    def is_obstacle_blocked(self, x, y, z, radius=0.55, feet_z=None, height=1.85):
        """
        Conservative AABB blocker used by both player and enemy movement.
        Prevents mobs walking through building walls/fences and keeps the
        camera/player body farther away from walls to avoid near-wall see-through.
        """
        if feet_z is None:
            feet_z = z
        for ox, oy, sx, sy, sz in self.obstacles:
            base_z = self.terrain_height(ox, oy)
            top_z = base_z + sz
            # If the body is clearly above the obstacle, do not block roof movement.
            if feet_z > top_z + 0.25:
                continue
            # If the obstacle is far above/below the body, ignore it.
            if feet_z + height < base_z + 0.15:
                continue
            if abs(x - ox) < sx / 2 + radius and abs(y - oy) < sy / 2 + radius:
                return True
        return False

    def segment_intersects_aabb(self, start, end, min_x, max_x, min_y, max_y, min_z, max_z):
        """
        Slab-method line segment vs AABB test.
        Used for bullet/enemy line-of-sight blocking by walls, fences, floors and roofs.
        """
        direction = end - start
        t_min = 0.0
        t_max = 1.0

        for axis in range(3):
            if axis == 0:
                s = start.x
                d = direction.x
                lo = min_x
                hi = max_x
            elif axis == 1:
                s = start.y
                d = direction.y
                lo = min_y
                hi = max_y
            else:
                s = start.z
                d = direction.z
                lo = min_z
                hi = max_z

            if abs(d) < 1e-7:
                if s < lo or s > hi:
                    return False
                continue

            inv = 1.0 / d
            t1 = (lo - s) * inv
            t2 = (hi - s) * inv

            if t1 > t2:
                t1, t2 = t2, t1

            t_min = max(t_min, t1)
            t_max = min(t_max, t2)

            if t_min > t_max:
                return False

        # Ignore contact exactly at the start/end to avoid self-blocking.
        return 0.015 < t_min < 0.985 or 0.015 < t_max < 0.985

    def has_line_of_sight(self, start, end):
        """
        True if a straight shot from start to end is not blocked by walls/fences/building roofs.

        This prevents enemies inside/below buildings from shooting through roofs or floors.
        It is deliberately conservative: if in doubt, the roof/wall blocks the shot.
        """
        start = Point3(start)
        end = Point3(end)

        # Vertical wall / fence / building blockers.
        for ox, oy, sx, sy, sz in self.obstacles:
            base_z = self.terrain_height(ox, oy)
            if self.segment_intersects_aabb(
                start, end,
                ox - sx / 2 - 0.06, ox + sx / 2 + 0.06,
                oy - sy / 2 - 0.06, oy + sy / 2 + 0.06,
                base_z - 0.05, base_z + sz + 0.12,
            ):
                return False

        # Horizontal blockers: roof / upper-floor walkable surfaces.
        # Movement can stand on these, but bullets/LOS should not pass through them.
        for surf in self.walkable_surfaces:
            kind = surf.get("kind", "")
            if kind not in ("roof", "flat", "upper_floor"):
                continue

            z = surf.get("z", None)
            if z is None:
                continue

            # Only matters if the shot crosses the roof/floor plane.
            dz0 = start.z - z
            dz1 = end.z - z
            if dz0 == 0 or dz1 == 0:
                continue
            if dz0 * dz1 > 0:
                continue

            t = (z - start.z) / max(1e-7, (end.z - start.z))
            if not (0.02 < t < 0.98):
                continue

            x = start.x + (end.x - start.x) * t
            y = start.y + (end.y - start.y) * t

            if abs(x - surf["x"]) <= surf["sx"] / 2 + 0.08 and abs(y - surf["y"]) <= surf["sy"] / 2 + 0.08:
                return False

        return True

    def resolve_actor_move(self, old_pos, wanted_pos, radius=0.55, height=1.85):
        """
        Axis-separated collision response for simple custom physics.
        This gives enemies/player a chance to slide along walls instead of
        passing through or getting stuck inside them.
        """
        feet_z = wanted_pos.z
        # Full move first.
        if not self.is_obstacle_blocked(wanted_pos.x, wanted_pos.y, wanted_pos.z, radius, feet_z, height):
            return Point3(wanted_pos)
        # Try X-only then Y-only slide.
        try_x = Point3(wanted_pos.x, old_pos.y, wanted_pos.z)
        if not self.is_obstacle_blocked(try_x.x, try_x.y, try_x.z, radius, feet_z, height):
            return try_x
        try_y = Point3(old_pos.x, wanted_pos.y, wanted_pos.z)
        if not self.is_obstacle_blocked(try_y.x, try_y.y, try_y.z, radius, feet_z, height):
            return try_y
        return Point3(old_pos)

    def add_roof_ladder(self, x, y, base_z, top_z, orientation="east"):
        """Add a visible climbable ladder on the outside of a building."""
        height = max(1.0, top_z - base_z)
        metal = (0.72, 0.66, 0.46, 1)
        dark = (0.18, 0.17, 0.14, 1)

        # Ladder plane is usually on the side of the building. Rungs are obvious bright metal.
        make_box(self.render, "ladder-left-rail", (0.08, 0.08, height), metal, (x, y - 0.42, base_z + height / 2))
        make_box(self.render, "ladder-right-rail", (0.08, 0.08, height), metal, (x, y + 0.42, base_z + height / 2))
        rung_count = max(5, int(height / 0.48))
        for i in range(rung_count + 1):
            z = base_z + 0.22 + i * (height - 0.44) / max(1, rung_count)
            make_box(self.render, "ladder-rung", (0.10, 0.98, 0.055), metal, (x, y, z))
        # Dark backing plate makes the ladder stand out against light buildings.
        make_box(self.render, "ladder-backing", (0.035, 1.15, height + 0.15), dark, (x + 0.035, y, base_z + height / 2))

        self.ladders.append({
            # Generous grab zone so the ladder is usable again.
            # The player is hard-snapped to the rails while climbing, so this does not reintroduce wall phasing.
            "x": x, "y": y, "sx": 2.25, "sy": 2.75,
            "z0": base_z, "z1": top_z + 0.2, "orientation": orientation,
        })

    def current_ladder(self, ignore_cooldown=False):
        if not ignore_cooldown and self.ladder_exit_cooldown > 0.0:
            return None
        # Detect ladders using the player's FEET height, not eye height.
        # V27 accidentally made the grab zone too precise; this makes ladders usable again.
        feet_z = self.player_pos.z - self.eye_height
        for ladder in self.ladders:
            dx = abs(self.player_pos.x - ladder["x"])
            dy = abs(self.player_pos.y - ladder["y"])
            if dx <= ladder["sx"] / 2 and dy <= ladder["sy"] / 2:
                if ladder["z0"] - 0.35 <= feet_z <= ladder["z1"] + 0.95:
                    return ladder
        return None

    def exit_ladder(self, ladder, top=None):
        """Snap the player safely off the ladder, either onto the roof or back onto the ground."""
        if top is None:
            mid = ladder["z0"] + (ladder["z1"] - ladder["z0"]) * 0.55 + self.eye_height
            top = self.player_pos.z >= mid

        # All current ladders are on the east/right side of buildings.
        # Top exit steps left onto the roof; bottom exit steps right away from the wall.
        if top:
            nx = ladder["x"] - 1.80
            ny = ladder["y"]
            nz = self.surface_ground_height(nx, ny, ladder["z1"] + self.eye_height) + self.eye_height
            self.player_pos = Point3(nx, ny, nz)
            self.show_message("Stepped onto roof", 0.8)
        else:
            nx = ladder["x"] + 1.65
            ny = ladder["y"]
            nz = self.terrain_height(nx, ny) + self.eye_height
            self.player_pos = Point3(nx, ny, nz)
            self.show_message("Stepped off ladder", 0.8)

        self.player_vel = Vec3(0, 0, 0)
        self.grounded = True
        # Prevent immediate re-grabbing of the same ladder.
        self.ladder_exit_cooldown = 0.55

    def create_terrain(self):
        length = self.level.length
        width = self.level.width
        step = 2.0
        xs = [x for x in self.frange(-width / 2, width / 2, step)]
        ys = [y for y in self.frange(-length / 2, length / 2 + 8, step)]
        fmt = GeomVertexFormat.getV3n3c4()
        vdata = GeomVertexData("level-terrain", fmt, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        color = GeomVertexWriter(vdata, "color")
        for y in ys:
            for x in xs:
                z = self.terrain_height(x, y)
                hx1 = self.terrain_height(x + 1, y)
                hx0 = self.terrain_height(x - 1, y)
                hy1 = self.terrain_height(x, y + 1)
                hy0 = self.terrain_height(x, y - 1)
                n = Vec3(hx0 - hx1, hy0 - hy1, 2.0)
                n.normalize()
                vertex.addData3(x, y, z)
                normal.addData3(n)
                if abs(x) < 6:
                    c = Vec4(0.36, 0.32, 0.24, 1)
                elif z < -0.25:
                    c = Vec4(0.29, 0.24, 0.17, 1)
                else:
                    c = Vec4(0.25, 0.38, 0.22, 1)
                color.addData4(c)
        tris = GeomTriangles(Geom.UHStatic)
        cols = len(xs)
        rows = len(ys)
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

    def add_maze_wall(self, x, y, sx, sy, h=2.05, color=(0.30, 0.31, 0.32, 1), name="maze-wall"):
        """Add a solid close-combat wall/fence section and register it for collision/bounces."""
        z = self.terrain_height(x, y) + h / 2
        make_box(self.render, name, (sx, sy, h), color, (x, y, z))
        self.obstacles.append((x, y, sx, sy, h))

    def add_fence(self, x, y, sx, sy, h=1.65):
        """Low dark fence/barrier, used to create winding lanes."""
        self.add_maze_wall(x, y, sx, sy, h, (0.16, 0.17, 0.16, 1), "fence")
        # Add a bright top rail so the maze is readable.
        if sx >= sy:
            make_box(self.render, "fence-top-rail", (sx, 0.14, 0.12), (0.48, 0.50, 0.47, 1), (x, y, self.terrain_height(x, y) + h + 0.06))
        else:
            make_box(self.render, "fence-top-rail", (0.14, sy, 0.12), (0.48, 0.50, 0.47, 1), (x, y, self.terrain_height(x, y) + h + 0.06))

    def create_two_floor_building(self, x, y, sx, sy, sz):
        """Closed two-storey building with more realistic visual detail and a side ladder to the roof."""
        base_z = self.terrain_height(x, y)
        wall_col = (0.62, 0.61, 0.58, 1)
        trim_col = (0.25, 0.26, 0.27, 1)
        roof_col = (0.20, 0.21, 0.22, 1)
        base_col = (0.40, 0.39, 0.36, 1)
        glass_col = (0.12, 0.16, 0.20, 1)
        thick = 0.42
        upper_z = base_z + sz * 0.52
        roof_z = base_z + sz + 0.22

        def wall(cx, cy, wx, wy, wz, label="twofloor-wall", color=None):
            make_box(self.render, label, (wx, wy, wz), color or wall_col, (cx, cy, base_z + wz / 2))
            self.obstacles.append((cx, cy, wx, wy, wz))

        make_box(self.render, "twofloor-foundation", (sx + 0.90, sy + 0.90, 0.36), base_col, (x, y, base_z + 0.18))
        make_box(self.render, "twofloor-floor", (sx, sy, 0.12), (0.34, 0.33, 0.30, 1), (x, y, base_z + 0.06))
        make_box(self.render, "twofloor-upper-floor", (sx - 0.8, sy - 0.8, 0.12), (0.28, 0.27, 0.24, 1), (x, y, upper_z))
        make_box(self.render, "twofloor-roof", (sx + 0.60, sy + 0.60, 0.24), roof_col, (x, y, roof_z))

        wall(x, y + sy / 2, sx, thick, sz)
        wall(x, y - sy / 2, sx, thick, sz)
        wall(x - sx / 2, y, thick, sy, sz)
        wall(x + sx / 2, y, thick, sy, sz)

        for cy in [y + sy / 2 + 0.02, y - sy / 2 - 0.02]:
            make_box(self.render, "facade-band", (sx + 0.12, 0.16, 0.48), base_col, (x, cy, base_z + 0.24))
        for cx in [x - sx / 2 - 0.02, x + sx / 2 + 0.02]:
            make_box(self.render, "facade-band-side", (0.16, sy + 0.12, 0.48), base_col, (cx, y, base_z + 0.24))
        for cx, cy in [
            (x - sx / 2, y - sy / 2), (x + sx / 2, y - sy / 2),
            (x - sx / 2, y + sy / 2), (x + sx / 2, y + sy / 2),
        ]:
            make_box(self.render, "corner-column", (0.34, 0.34, sz + 0.08), trim_col, (cx, cy, base_z + (sz + 0.08) / 2))

        wall(x, y + sy / 2, sx + 0.36, 0.24, 0.75, "roof-parapet", trim_col)
        wall(x - sx / 2, y, 0.24, sy + 0.36, 0.75, "roof-parapet", trim_col)
        wall(x + sx / 2, y, 0.24, sy + 0.36, 0.75, "roof-parapet", trim_col)
        wall(x, y - sy / 2, sx + 0.36, 0.24, 0.75, "roof-parapet", trim_col)

        for side_y in [y + sy / 2 + 0.06, y - sy / 2 - 0.06]:
            for wx in [-0.36, -0.12, 0.12, 0.36]:
                for hz in [1.22, sz * 0.72]:
                    make_box(self.render, "window-trim", (sx * 0.095, 0.12, 0.68), trim_col, (x + wx * sx, side_y, base_z + hz))
                    make_box(self.render, "window-sill", (sx * 0.105, 0.10, 0.08), base_col, (x + wx * sx, side_y, base_z + hz - 0.36))
                    make_box(self.render, "window-pane", (sx * 0.078, 0.08, 0.56), glass_col, (x + wx * sx, side_y + (0.02 if side_y > y else -0.02), base_z + hz))
        for side_x in [x - sx / 2 - 0.06, x + sx / 2 + 0.06]:
            for wy in [-0.24, 0.24]:
                for hz in [1.30, sz * 0.74]:
                    make_box(self.render, "side-window-trim", (0.12, sy * 0.16, 0.62), trim_col, (side_x, y + wy * sy, base_z + hz))
                    make_box(self.render, "side-window-pane", (0.08, sy * 0.135, 0.52), glass_col, (side_x + (0.02 if side_x > x else -0.02), y + wy * sy, base_z + hz))

        ladder_x = x + sx / 2 + 0.58
        ladder_y = y - sy * 0.18
        self.add_roof_ladder(ladder_x, ladder_y, base_z + 0.08, roof_z + 0.12, orientation="east")
        self.add_walkable_surface("roof", x, y, sx - 0.85, sy - 0.85, z=roof_z + 0.12, min_z=roof_z - 1.0)

        make_box(self.render, "roof-vent", (1.55, 1.05, 0.62), (0.10, 0.11, 0.11, 1), (x - sx * 0.28, y + sy * 0.18, roof_z + 0.31))
        make_box(self.render, "water-tank", (1.05, 1.05, 1.08), (0.18, 0.20, 0.22, 1), (x + sx * 0.12, y + sy * 0.08, roof_z + 0.54))
        make_box(self.render, "rooftop-crate", (1.18, 0.92, 0.78), (0.42, 0.29, 0.16, 1), (x - sx * 0.16, y - sy * 0.12, roof_z + 0.40))
        make_box(self.render, "roof-ac", (1.25, 0.90, 0.60), (0.22, 0.23, 0.24, 1), (x + sx * 0.24, y - sy * 0.18, roof_z + 0.30))
        for px, py in [(x - sx / 2 - 0.16, y - sy * 0.32), (x + sx / 2 + 0.16, y + sy * 0.28)]:
            make_box(self.render, "drainpipe", (0.10, 0.10, sz + 0.30), (0.22, 0.22, 0.22, 1), (px, py, base_z + (sz + 0.30) / 2))

    def create_level_one_maze(self):
        """Hand-built close-combat maze inspired by the user's sketch."""
        # Spawn motor pool markers and low barriers.
        self.add_fence(-10, -59, 14, 0.55, 1.15)
        self.add_fence(10, -59, 14, 0.55, 1.15)
        self.add_fence(-32, -43, 0.8, 28, 1.75)
        self.add_fence(32, -43, 0.8, 28, 1.75)

        # Big two-floor building near the start with a clear front door.
        self.create_two_floor_building(-14, -18, 18, 18, 5.2)

        # Winding fence lanes: deliberately block long cross-map lines of sight.
        maze_segments = [
            (12, -38, 0.75, 26), (25, -27, 15, 0.75),
            (7, -12, 18, 0.75), (24, -3, 0.75, 28),
            (-28, 10, 20, 0.75), (-17, 18, 0.75, 22),
            (1, 28, 24, 0.75), (15, 38, 0.75, 22),
            (-26, 50, 18, 0.75), (-5, 59, 0.75, 17),
            (25, 55, 16, 0.75),
        ]
        for x, y, sx, sy in maze_segments:
            self.add_fence(x, y, sx, sy, 1.75)

        # Heavier concrete blocks around corners, forcing close-range entry fights.
        concrete = [
            (5, -46, 5, 1.3, 1.6), (-26, -31, 1.4, 6, 1.6),
            (27, -13, 1.3, 8, 1.6), (-4, 4, 7, 1.2, 1.4),
            (-29, 30, 1.4, 7, 1.5), (22, 23, 7, 1.2, 1.5),
            (2, 46, 8, 1.2, 1.5), (-21, 61, 6, 1.2, 1.5),
        ]
        for x, y, sx, sy, h in concrete:
            self.add_maze_wall(x, y, sx, sy, h, (0.38, 0.38, 0.37, 1), "concrete-maze-block")

        # Small guard huts / rooms around spawners.
        self.create_enterable_building(23, 14, 8, 8, 3.0)
        self.create_enterable_building(-23, 42, 8, 8, 3.0)
        self.create_enterable_building(21, 52, 8, 7, 3.0)

    def create_close_combat_maze_for_level(self):
        """Create a close-combat, maze-like layout for every level.

        Level 1 keeps the hand-built layout based on the user's sketch.
        Later levels use a scaled winding pattern: staggered fences, concrete
        blast walls, enterable two-floor buildings, spawner rooms, and short
        sightlines. This makes every level play like a CQB raid rather than
        an open-field sniper map.
        """
        if self.level_index == 0:
            self.create_level_one_maze()
            return

        L = self.level.length
        W = self.level.width
        idx = self.level_index
        start_y = -L / 2 + 12
        exit_y = L / 2 - 8
        difficulty = idx + 1

        # Motor pool / player start area, with friendly vehicles protected by low barriers.
        self.add_fence(-W * 0.24, start_y + 4, W * 0.22, 0.55, 1.10)
        self.add_fence(W * 0.24, start_y + 4, W * 0.22, 0.55, 1.10)
        self.add_fence(-W * 0.43, start_y + 15, 0.75, 22, 1.55)
        self.add_fence(W * 0.43, start_y + 15, 0.75, 22, 1.55)

        # Alternating horizontal barriers: each row leaves a gap on alternating sides,
        # forcing a winding route and blocking cross-map lines of sight.
        gap_width = max(12.0, W * 0.18)
        y = start_y + 26
        row = 0
        while y < exit_y - 12:
            gap_center = ((-1) ** row) * W * (0.23 + 0.03 * (row % 2))
            left_bound = -W / 2 + 5
            right_bound = W / 2 - 5
            left_end = gap_center - gap_width / 2
            right_start = gap_center + gap_width / 2

            if left_end - left_bound > 5:
                self.add_fence((left_bound + left_end) / 2, y, left_end - left_bound, 0.70, 1.70)
            if right_bound - right_start > 5:
                self.add_fence((right_start + right_bound) / 2, y, right_bound - right_start, 0.70, 1.70)

            # Concrete anchors at the ends of each barrier make better close cover.
            self.add_maze_wall(left_bound + 3.5, y + 1.2, 4.2, 1.15, 1.55, (0.39, 0.39, 0.38, 1), "maze-anchor")
            self.add_maze_wall(right_bound - 3.5, y - 1.2, 4.2, 1.15, 1.55, (0.39, 0.39, 0.38, 1), "maze-anchor")

            # Short vertical fences behind gaps so the path snakes rather than becomes a straight lane.
            v_x = gap_center + ((-1) ** (row + 1)) * min(8, W * 0.11)
            self.add_fence(v_x, y + 8.5, 0.72, 13.5, 1.65)

            y += max(18.0, 24.0 - min(8, difficulty))
            row += 1

        # Large enterable buildings: more of them as difficulty rises.
        building_count = min(5, 2 + idx)
        usable_span = exit_y - start_y - 34
        for b in range(building_count):
            by = start_y + 30 + usable_span * (b + 0.5) / building_count
            side = -1 if b % 2 == 0 else 1
            bx = side * W * (0.20 + 0.05 * ((b + idx) % 2))
            bsx = clamp(14 + difficulty * 1.5, 14, 22)
            bsy = clamp(14 + (b % 2) * 3 + difficulty, 14, 22)
            bsz = clamp(4.6 + 0.25 * difficulty, 4.8, 6.2)
            if b == 0:
                self.create_two_floor_building(bx, by, bsx, bsy, bsz)
            else:
                self.create_enterable_building(bx, by, bsx * 0.75, bsy * 0.72, min(4.2, bsz))

        # Spawner bunkers: make each spawner feel like a room/corner fight instead of
        # an exposed target in a field.
        for i, sp in enumerate(self.level.spawners):
            sx, sy = sp[0], sp[1]
            side = -1 if i % 2 == 0 else 1
            # U-shaped small bunker around the spawner, with one open side.
            self.add_maze_wall(sx, sy + 4.2, 8.0, 0.8, 1.55, (0.31, 0.31, 0.30, 1), "spawner-bunker-wall")
            self.add_maze_wall(sx - side * 4.2, sy, 0.8, 7.6, 1.55, (0.31, 0.31, 0.30, 1), "spawner-bunker-wall")
            self.add_maze_wall(sx + side * 2.6, sy - 3.4, 3.2, 0.8, 1.35, (0.31, 0.31, 0.30, 1), "spawner-bunker-wall")
            # Nearby barrel or crate cover for tactical choices.
            if i % 2 == 0:
                bx, by = sx + side * 5.0, sy + 1.5
                make_box(self.render, "spawner-crate", (1.5, 1.1, 0.85), (0.43, 0.30, 0.16, 1), (bx, by, self.terrain_height(bx, by) + 0.43))
                self.obstacles.append((bx, by, 1.5, 1.1, 0.85))

        # Final extraction approach: still green and visible, but protected by offset gates.
        self.add_fence(-W * 0.23, exit_y - 13, W * 0.36, 0.65, 1.55)
        self.add_fence(W * 0.28, exit_y - 5, W * 0.32, 0.65, 1.55)
        self.add_maze_wall(-W * 0.39, exit_y - 7, 0.8, 13, 1.65, (0.25, 0.27, 0.26, 1), "exit-gate-wall")
        self.add_maze_wall(W * 0.39, exit_y - 13, 0.8, 13, 1.65, (0.25, 0.27, 0.26, 1), "exit-gate-wall")

    def create_level_objects(self):
        # Boundaries and outpost/depot geometry.
        length = self.level.length
        width = self.level.width
        wall_mat = make_material((0.34, 0.34, 0.34, 1))
        for side in [-1, 1]:
            x = side * (width / 2 + 1.5)
            make_box(self.render, "boundary-wall", (2.0, length + 10, 3.4), (0.25, 0.26, 0.27, 1), (x, 0, 1.7), material=wall_mat)
            self.obstacles.append((x, 0, 2.0, length + 10, 3.4))

        # Extraction zone.
        exit_y = length / 2 - 8
        self.exit_pos = Point3(0, exit_y, self.terrain_height(0, exit_y))
        self.exit_node = self.render.attachNewNode("extraction-zone")
        self.exit_node.setPos(self.exit_pos)
        make_box(self.exit_node, "pad", (10.0, 5.5, 0.14), (0.05, 1.0, 0.15, 0.88), (0, 0, 0.07))
        label = TextNode("exit-label")
        label.setText("EXTRACTION")
        label.setAlign(TextNode.ACenter)
        label.setTextColor(0.1, 1.0, 0.2, 1)
        lab_np = self.exit_node.attachNewNode(label)
        lab_np.setScale(0.55)
        lab_np.setPos(0, 0, 2.0)

        # Every level gets a close-combat maze layout.
        self.create_close_combat_maze_for_level()

        # Cover/obstacles.
        for x, y, sx, sy, sz in self.level.obstacles:
            z = self.terrain_height(x, y) + sz / 2
            make_box(self.render, "concrete-cover", (sx, sy, sz), (0.36, 0.36, 0.36, 1), (x, y, z))
            self.obstacles.append((x, y, sx, sy, sz))

        # Extra level geometry: low buildings, alleys, gates and depot blocks.
        # These make the levels less like a flat corridor and more like a 3D raid map.
        building_sets = [
            [(-18, -32, 8, 7, 3.2), (18, -26, 7, 8, 3.0), (-24, 35, 9, 8, 3.4), (24, 56, 8, 10, 3.2)],
            [(-26, -2, 7, 12, 3.0), (25, 18, 8, 9, 3.4), (-23, 54, 10, 9, 3.2), (22, 74, 9, 11, 3.6)],
            [(-27, -25, 9, 12, 3.6), (26, -5, 10, 10, 3.4), (-26, 34, 11, 11, 3.8), (24, 62, 12, 10, 3.6), (-14, 95, 10, 12, 4.0), (14, 108, 12, 9, 4.0)],
        ]
        for x, y, sx, sy, sz in building_sets[min(self.level_index, 2)]:
            self.create_enterable_building(x, y, sx, sy, sz)

        # Enemy placements: only a patrol screen is present initially.
        # More enemies come from visible spawner nodes when the player approaches.
        for kind, x, y in self.level.enemies:
            self.enemies.append(Enemy(self, (x, y, self.terrain_height(x, y)), kind))
        for x, y, kinds, total, interval, radius in self.level.spawners:
            self.spawners.append(MobSpawner(self, (x, y, self.terrain_height(x, y)), kinds, total, interval, radius))
        for x, y, h in self.level.tanks:
            self.enemy_tanks.append(EnemyTank(self, (x, y, self.terrain_height(x, y)), h))

        # Objectives.
        if self.level.required_destroy > 0:
            if self.level_index == 0:
                self.destructibles.append(Destructible(self, "radio", (5, self.level.length / 2 - 28, self.terrain_height(5, self.level.length / 2 - 28)), 140))
            elif self.level_index == 1:
                self.destructibles.append(Destructible(self, "generator", (-6, self.level.length / 2 - 26, self.terrain_height(-6, self.level.length / 2 - 26)), 170))
            else:
                self.destructibles.append(Destructible(self, "radio", (-10, self.level.length / 2 - 35, self.terrain_height(-10, self.level.length / 2 - 35)), 155))
                self.destructibles.append(Destructible(self, "generator", (12, self.level.length / 2 - 22, self.terrain_height(12, self.level.length / 2 - 22)), 190))

        for x, y in self.level.hostages:
            self.hostages.append(Hostage(self, (x, y, self.terrain_height(x, y))))
        for kind, x, y in self.level.pickups:
            self.pickups.append(Pickup(self, kind, (x, y, self.terrain_height(x, y) + 0.25)))
        for x, y in self.level.explosive_barrels:
            barrel = Destructible(self, "barrel", (x, y, self.terrain_height(x, y)), 45)
            barrel.radius = 0.75
            barrel.node.removeNode()
            barrel.node = self.render.attachNewNode("explosive-barrel")
            barrel.node.setPos(barrel.pos)
            make_cylinder_approx(barrel.node, "barrel-body", 0.42, 0.95, (0.65, 0.08, 0.04, 1), (0, 0, 0.55), (90, 0, 0), 18)
            self.explosive_barrels.append(barrel)

        if self.level.apc_pos:
            x, y = self.level.apc_pos
            self.apc = FriendlyAPC(self, (x, y, self.terrain_height(x, y)))

        if self.level.friendly_tank_pos:
            x, y = self.level.friendly_tank_pos
            self.friendly_tank = FriendlyTank(self, (x, y, self.terrain_height(x, y)))

    def create_enterable_building(self, x, y, sx, sy, sz):
        """Closed one-storey building with a side ladder and more realistic visual detail."""
        base_z = self.terrain_height(x, y)
        wall_col = (0.60, 0.58, 0.54, 1)
        trim_col = (0.24, 0.24, 0.24, 1)
        roof_col = (0.20, 0.20, 0.21, 1)
        base_col = (0.39, 0.37, 0.34, 1)
        glass_col = (0.11, 0.15, 0.18, 1)
        thick = 0.38
        roof_z = base_z + sz + 0.12

        def wall(cx, cy, wx, wy, wz, name="building-wall", color=None):
            zc = base_z + wz / 2
            make_box(self.render, name, (wx, wy, wz), color or wall_col, (cx, cy, zc))
            self.obstacles.append((cx, cy, wx, wy, wz))

        make_box(self.render, "building-foundation", (sx + 0.70, sy + 0.70, 0.30), base_col, (x, y, base_z + 0.15))
        make_box(self.render, "building-floor", (sx, sy, 0.10), (0.31, 0.30, 0.28, 1), (x, y, base_z + 0.05))
        make_box(self.render, "building-roof", (sx + 0.32, sy + 0.32, 0.22), roof_col, (x, y, roof_z))

        wall(x, y + sy / 2, sx, thick, sz)
        wall(x, y - sy / 2, sx, thick, sz)
        wall(x - sx / 2, y, thick, sy, sz)
        wall(x + sx / 2, y, thick, sy, sz)

        for cy in [y + sy / 2 + 0.02, y - sy / 2 - 0.02]:
            make_box(self.render, "building-base-band", (sx + 0.08, 0.14, 0.42), base_col, (x, cy, base_z + 0.21))
        for cx, cy, wx, wy in [(x, y + sy / 2, sx + 0.30, 0.18), (x, y - sy / 2, sx + 0.30, 0.18), (x - sx / 2, y, 0.18, sy + 0.30), (x + sx / 2, y, 0.18, sy + 0.30)]:
            make_box(self.render, "roof-edge", (wx, wy, 0.42), trim_col, (cx, cy, roof_z + 0.19))

        for side_y in [y + sy / 2 + 0.05, y - sy / 2 - 0.05]:
            for wx in [-0.30, 0.30]:
                make_box(self.render, "window-trim", (sx * 0.17, 0.11, 0.66), trim_col, (x + wx * sx, side_y, base_z + 1.72))
                make_box(self.render, "window-sill", (sx * 0.18, 0.10, 0.08), base_col, (x + wx * sx, side_y, base_z + 1.38))
                make_box(self.render, "window-pane", (sx * 0.145, 0.08, 0.54), glass_col, (x + wx * sx, side_y, base_z + 1.72))
        for side_x in [x - sx / 2 - 0.05, x + sx / 2 + 0.05]:
            make_box(self.render, "side-window-trim", (0.11, sy * 0.22, 0.64), trim_col, (side_x, y, base_z + 1.72))
            make_box(self.render, "side-window-pane", (0.08, sy * 0.19, 0.52), glass_col, (side_x + (0.02 if side_x > x else -0.02), y, base_z + 1.72))

        ladder_x = x + sx / 2 + 0.52
        ladder_y = y - sy * 0.10
        self.add_roof_ladder(ladder_x, ladder_y, base_z + 0.08, roof_z + 0.10, orientation="east")
        self.add_walkable_surface("roof", x, y, sx - 0.8, sy - 0.8, z=roof_z + 0.10, min_z=roof_z - 0.9)

        make_box(self.render, "roof-box", (1.15, 0.90, 0.58), (0.14, 0.15, 0.15, 1), (x - sx * 0.15, y + sy * 0.15, roof_z + 0.29))
        make_box(self.render, "roof-duct", (1.30, 0.65, 0.36), (0.18, 0.19, 0.19, 1), (x + sx * 0.18, y - sy * 0.10, roof_z + 0.18))
        make_box(self.render, "meter-box", (0.34, 0.24, 0.58), (0.25, 0.25, 0.25, 1), (x - sx / 2 - 0.20, y + sy * 0.18, base_z + 0.30))

    @staticmethod
    def frange(start, stop, step):
        x = start
        while x <= stop:
            yield x
            x += step

    # ------------------------------------------------------------
    # Input and state
    # ------------------------------------------------------------

    def set_key(self, key, value):
        self.keys[key] = value

    def set_mouse(self, value):
        self.mouse_down = value

    def toggle_pause(self):
        self.paused = not self.paused
        self.hud_pause.setText("PAUSED\nC Continue   R Restart Level   M Mouse Lock   Q Quit" if self.paused else "")
        if self.paused:
            self.release_mouse()
        else:
            self.lock_mouse()

    def continue_from_pause(self):
        if self.paused:
            self.paused = False
            self.hud_pause.setText("")
            self.lock_mouse()

    def cancel_or_continue(self):
        if self.mortar_targeting:
            self.cancel_mortar_targeting()
        elif self.paused:
            self.continue_from_pause()

    def switch_tank_ammo(self):
        if self.in_tank and self.friendly_tank:
            self.friendly_tank.switch_ammo()

    def mortar_action(self):
        if self.paused:
            return
        if not self.mortar_targeting:
            if self.money < self.mortar_cost:
                self.show_message(f"Need ${self.mortar_cost} for mortar strike")
                return
            if self.mortar_cooldown > 0:
                self.show_message(f"Mortar reloading: {self.mortar_cooldown:.0f}s")
                return
            # Start target on the point you are looking toward, then adjust it on minimap with arrows.
            direction = self.camera.getQuat(self.render).getForward()
            direction.z = 0
            if direction.lengthSquared() <= 0.001:
                direction = Vec3(0, 1, 0)
            direction.normalize()
            self.mortar_target = Point3(
                clamp(self.player_pos.x + direction.x * 34, -self.level.width / 2 + 4, self.level.width / 2 - 4),
                clamp(self.player_pos.y + direction.y * 34, -self.level.length / 2 + 4, self.level.length / 2 - 4),
                0,
            )
            self.mortar_target.z = self.terrain_height(self.mortar_target.x, self.mortar_target.y)
            self.mortar_targeting = True
            self.show_message("ARTILLERY SIGHT: arrows slew target | X/Enter fire | C cancel", 2.6)
        else:
            self.money -= self.mortar_cost
            self.mortar_cooldown = 20.0
            self.mortar_pending.append({"pos": Point3(self.mortar_target), "timer": 1.6})
            self.mortar_targeting = False
            if self.mortar_marker_np:
                self.mortar_marker_np.removeNode()
                self.mortar_marker_np = None
            self.show_message("Mortar strike inbound", 1.6)

    def cancel_mortar_targeting(self):
        self.mortar_targeting = False
        if self.mortar_marker_np:
            self.mortar_marker_np.removeNode()
            self.mortar_marker_np = None
        self.show_message("Mortar cancelled", 1.0)

    def update_mortar(self, dt):
        self.mortar_cooldown = max(0.0, self.mortar_cooldown - dt)
        if self.mortar_targeting:
            move = 18.0 * dt
            # Arrow keys move the target on the minimap/world.
            if self.mouseWatcherNode.isButtonDown("arrow_left"):
                self.mortar_target.x -= move
            if self.mouseWatcherNode.isButtonDown("arrow_right"):
                self.mortar_target.x += move
            if self.mouseWatcherNode.isButtonDown("arrow_up"):
                self.mortar_target.y += move
            if self.mouseWatcherNode.isButtonDown("arrow_down"):
                self.mortar_target.y -= move
            self.mortar_target.x = clamp(self.mortar_target.x, -self.level.width / 2 + 4, self.level.width / 2 - 4)
            self.mortar_target.y = clamp(self.mortar_target.y, -self.level.length / 2 + 4, self.level.length / 2 - 4)
            self.mortar_target.z = self.terrain_height(self.mortar_target.x, self.mortar_target.y)
            self.draw_mortar_marker()
        remaining = []
        for strike in self.mortar_pending:
            strike["timer"] -= dt
            if strike["timer"] <= 0:
                pos = strike["pos"]
                # Three-round walking impact pattern.
                for ox, oy in [(0, 0), (2.2, -1.4), (-2.0, 1.5)]:
                    p = Point3(pos.x + ox, pos.y + oy, self.terrain_height(pos.x + ox, pos.y + oy) + 0.1)
                    self.spawn_explosion(p, radius=7.8, damage=145, source="mortar")
            else:
                remaining.append(strike)
        self.mortar_pending = remaining

    def draw_mortar_marker(self):
        if self.mortar_marker_np:
            self.mortar_marker_np.removeNode()
        root = self.render.attachNewNode("mortar-target-marker")
        root.setPos(self.mortar_target)
        self.mortar_marker_np = root

        # War-Thunder-style artillery aiming marker:
        # a ground impact reticle with range brackets and dispersion rings.
        seg = LineSegs("mortar-artillery-sight")
        seg.setThickness(3)
        seg.setColor(1.0, 0.12, 0.04, 1)
        r = 2.2
        seg.moveTo(-r, 0, 0.10); seg.drawTo(-0.55, 0, 0.10)
        seg.moveTo(0.55, 0, 0.10); seg.drawTo(r, 0, 0.10)
        seg.moveTo(0, -r, 0.10); seg.drawTo(0, -0.55, 0.10)
        seg.moveTo(0, 0.55, 0.10); seg.drawTo(0, r, 0.10)

        # Bracket square.
        b = 2.9
        gap = 1.5
        # four corner brackets
        seg.moveTo(-b, -b, 0.10); seg.drawTo(-gap, -b, 0.10); seg.moveTo(-b, -b, 0.10); seg.drawTo(-b, -gap, 0.10)
        seg.moveTo(b, -b, 0.10); seg.drawTo(gap, -b, 0.10); seg.moveTo(b, -b, 0.10); seg.drawTo(b, -gap, 0.10)
        seg.moveTo(-b, b, 0.10); seg.drawTo(-gap, b, 0.10); seg.moveTo(-b, b, 0.10); seg.drawTo(-b, gap, 0.10)
        seg.moveTo(b, b, 0.10); seg.drawTo(gap, b, 0.10); seg.moveTo(b, b, 0.10); seg.drawTo(b, gap, 0.10)

        # Dispersion rings.
        for rad in [3.0, 5.2, 7.8]:
            first = True
            for i in range(65):
                a = math.tau * i / 64
                px, py = math.cos(a) * rad, math.sin(a) * rad
                if first:
                    seg.moveTo(px, py, 0.09)
                    first = False
                else:
                    seg.drawTo(px, py, 0.09)

        root.attachNewNode(seg.create())

        # Small vertical spotting mast so the target is visible from a distance.
        make_box(root, "mortar-spotting-mast", (0.10, 0.10, 2.2), (1.0, 0.08, 0.04, 0.75), (0, 0, 1.1))

    def toggle_mouse_free(self):
        if not self.paused:
            return
        if self.mouse_free:
            self.lock_mouse()
        else:
            self.release_mouse()

    def release_mouse(self):
        props = WindowProperties()
        props.setCursorHidden(False)
        self.win.requestProperties(props)
        self.mouse_free = True

    def lock_mouse(self):
        props = WindowProperties()
        props.setCursorHidden(True)
        self.win.requestProperties(props)
        self.center_x = self.win.getXSize() // 2
        self.center_y = self.win.getYSize() // 2
        self.win.movePointer(0, self.center_x, self.center_y)
        self.mouse_free = False

    def switch_weapon(self, idx):
        if 0 <= idx < len(self.weapons) and not self.in_apc:
            self.current_weapon = idx
            self.reloading = False
            self.reload_timer = 0
            self.reload_total = 0

    def toggle_scope(self):
        if self.paused:
            return
        self.scoped = not self.scoped
        self.cam.node().getLens().setFov(self.scope_fov if self.scoped else self.normal_fov)
        self.mouse_sensitivity = 0.055 if self.scoped else 0.12
        self.draw_scope_overlay()

    def reload_or_vehicle_reload(self):
        if self.in_apc and self.apc:
            if self.apc.ammo < self.apc.mag_size and self.apc.reserve > 0:
                self.apc.reload_timer = 3.2
            return
        self.start_reload()

    def start_reload(self):
        if self.reloading:
            return
        w = self.weapons[self.current_weapon]
        if self.magazines[self.current_weapon] >= w.mag_size or self.reserves[self.current_weapon] <= 0:
            return
        self.reloading = True
        self.reload_timer = w.reload_time
        self.reload_total = w.reload_time

    def finish_reload(self):
        w = self.weapons[self.current_weapon]
        need = w.mag_size - self.magazines[self.current_weapon]
        take = min(need, self.reserves[self.current_weapon])
        self.magazines[self.current_weapon] += take
        self.reserves[self.current_weapon] -= take
        self.reloading = False
        self.reload_timer = 0
        self.reload_total = 0

    def interact(self):
        if self.paused:
            return
        ladder = self.current_ladder(ignore_cooldown=True)
        if ladder:
            self.exit_ladder(ladder)
            return
        if self.in_tank and self.friendly_tank:
            self.in_tank = False
            self.friendly_tank.occupied = False
            self.player_pos = Point3(self.friendly_tank.pos.x + 3.0, self.friendly_tank.pos.y, self.friendly_tank.pos.z + self.eye_height)
            self.show_message("Exited tank")
            return
        if self.in_apc:
            self.in_apc = False
            self.apc.occupied = False
            self.player_pos = Point3(self.apc.pos.x + 2.8, self.apc.pos.y, self.apc.pos.z + self.eye_height)
            self.show_message("Exited APC")
            return
        if self.friendly_tank and (Vec3(self.friendly_tank.pos.x - self.player_pos.x, self.friendly_tank.pos.y - self.player_pos.y, 0)).length() < 4.6:
            self.in_tank = True
            self.friendly_tank.occupied = True
            self.show_message("Entered tank: mouse aims, LMB fires cannon")
            return
        if self.apc and (Vec3(self.apc.pos.x - self.player_pos.x, self.apc.pos.y - self.player_pos.y, 0)).length() < 4.0:
            self.in_apc = True
            self.apc.occupied = True
            self.show_message("Entered APC: Space fires LMG")
            return
        # Rescue hostage if nearby.
        for hostage in self.hostages:
            if not hostage.rescued and (hostage.pos - self.player_pos).length() < 3.2:
                hostage.rescued = True
                hostage.node.removeNode()
                self.rescued_hostages += 1
                self.money += 75
                self.show_message("Hostage rescued")
                return
        # Extraction is automatic now: just walk into the bright green zone.

    def use_nearby_pickup(self):
        for p in self.pickups:
            if not p.taken and (p.pos - self.player_pos).length() < 3.0:
                p.taken = True
                p.node.removeNode()
                if p.kind == "health":
                    self.health = min(self.max_health, self.health + 45)
                    self.show_message("Medkit used")
                else:
                    for i, w in enumerate(self.weapons):
                        self.reserves[i] += w.mag_size * 2
                    self.grenades += 2
                    self.show_message("Ammo restocked")
                return

    # ------------------------------------------------------------
    # Movement, camera, combat target
    # ------------------------------------------------------------

    def update_mouse_look(self):
        if self.paused or self.mouse_free:
            return
        if not self.mouseWatcherNode.hasMouse():
            return
        pointer = self.win.getPointer(0)
        dx = pointer.getX() - self.center_x
        dy = pointer.getY() - self.center_y
        self.yaw -= dx * self.mouse_sensitivity
        self.pitch -= dy * self.mouse_sensitivity
        self.pitch = clamp(self.pitch, -84, 84)
        self.win.movePointer(0, self.center_x, self.center_y)

    def update_player(self, dt):
        # Mouse look is processed exactly once per frame here.
        # WASD is camera-relative, using Panda3D's heading convention:
        # looking left then pressing W moves left.
        self.update_mouse_look()
        self.ladder_exit_cooldown = max(0.0, self.ladder_exit_cooldown - dt)
        if self.in_tank and self.friendly_tank:
            self.friendly_tank.update(dt)
            return
        if self.in_apc and self.apc:
            self.apc.update(dt)
            return
        yaw_rad = math.radians(self.yaw)
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
        if wish.lengthSquared() > 0:
            wish.normalize()
        speed = 5.1
        if self.scoped:
            speed *= 0.55
        self.player_vel.x = wish.x * speed
        self.player_vel.y = wish.y * speed

        ladder = self.current_ladder()
        if ladder:
            # Ladder climbing is a locked state: no horizontal WASD drift.
            # The grab zone is generous, but once attached you are snapped to the rails.
            climb = 0.0
            if self.keys["w"] or self.keys["space"]:
                climb += 4.1
            if self.keys["s"]:
                climb -= 3.4
            self.player_vel.x = 0.0
            self.player_vel.y = 0.0
            self.player_vel.z = climb
            self.grounded = False
            self.player_pos.x = ladder["x"]
            self.player_pos.y = ladder["y"]
            # If the player enters the ladder zone from the ground, start at the bottom of the ladder.
            min_eye_z = ladder["z0"] + self.eye_height + 0.02
            if self.player_pos.z < min_eye_z:
                self.player_pos.z = min_eye_z
        else:
            ground = self.surface_ground_height(self.player_pos.x, self.player_pos.y, self.player_pos.z) + self.eye_height
            if self.player_pos.z <= ground + 0.03:
                self.grounded = True
                self.player_pos.z = ground
                self.player_vel.z = max(0, self.player_vel.z)
            else:
                self.grounded = False
            if self.keys["space"] and self.grounded:
                self.player_vel.z = 5.3
                self.grounded = False
            self.player_vel.z -= 9.81 * dt

        old_pos = Point3(self.player_pos)
        self.player_pos += self.player_vel * dt
        if ladder:
            top_z = ladder["z1"] + self.eye_height
            bottom_z = ladder["z0"] + self.eye_height
            if self.player_pos.z >= top_z - 0.06:
                # Auto-exit onto the roof when you reach the top.
                self.exit_ladder(ladder, top=True)
                ladder = None
            elif self.player_pos.z <= bottom_z + 0.06 and self.keys["s"]:
                # Climb down and step away from the wall.
                self.exit_ladder(ladder, top=False)
                ladder = None
            else:
                self.player_pos.z = clamp(self.player_pos.z, bottom_z, top_z)
        # bounds
        self.player_pos.x = clamp(self.player_pos.x, -self.level.width / 2 + 1.6, self.level.width / 2 - 1.6)
        self.player_pos.y = clamp(self.player_pos.y, -self.level.length / 2 + 2, self.level.length / 2 - 2)
        # obstacle collision with a larger camera/body safety margin.
        # This prevents the camera from getting inside walls, which causes the
        # classic close-up see-through/clipping artefact. Do not apply it while
        # actively attached to a ladder, because the ladder deliberately sits on
        # a wall face.
        if not ladder:
            moved = self.resolve_actor_move(old_pos, self.player_pos, radius=0.92, height=self.eye_height)
            if moved.x != self.player_pos.x or moved.y != self.player_pos.y:
                self.player_pos.x = moved.x
                self.player_pos.y = moved.y
                self.player_vel.x *= 0.15
                self.player_vel.y *= 0.15
        ground2 = self.surface_ground_height(self.player_pos.x, self.player_pos.y, self.player_pos.z) + self.eye_height
        if self.player_pos.z < ground2:
            self.player_pos.z = ground2
            self.player_vel.z = 0
            self.grounded = True
        self.camera.setPos(self.player_pos)
        self.camera.setHpr(self.yaw, self.pitch, 0)

    def get_combat_target_pos(self):
        if self.in_tank and self.friendly_tank:
            return Point3(self.friendly_tank.pos.x, self.friendly_tank.pos.y, self.friendly_tank.pos.z + 1.55)
        if self.in_apc and self.apc:
            return Point3(self.apc.pos.x, self.apc.pos.y, self.apc.pos.z + 1.3)
        return Point3(self.player_pos.x, self.player_pos.y, self.player_pos.z - self.eye_height + 0.9)

    # ------------------------------------------------------------
    # Weapons/projectiles
    # ------------------------------------------------------------

    def try_fire(self):
        if self.paused:
            return
        if self.in_tank and self.friendly_tank:
            self.friendly_tank.fire_cannon()
            return
        if self.in_apc and self.apc:
            # APC left click fires LMG too.
            self.apc.fire_lmg()
            return
        if self.reloading:
            return
        w = self.weapons[self.current_weapon]
        if self.magazines[self.current_weapon] <= 0:
            self.start_reload()
            return
        if self.fire_timer > 0:
            return
        self.magazines[self.current_weapon] -= 1
        self.fire_timer = 60.0 / w.rpm
        for _ in range(w.pellets):
            self.fire_weapon_projectile(w)

    def fire_weapon_projectile(self, weapon):
        cam_pos = self.camera.getPos(self.render)
        quat = self.camera.getQuat(self.render)
        direction = quat.getForward()
        right = quat.getRight()
        up = quat.getUp()
        spread = math.radians(weapon.spread_deg * (0.35 if self.scoped else 1.0))
        direction = direction + right * random.gauss(0, spread) + up * random.gauss(0, spread)
        direction.normalize()
        start = cam_pos + direction * 0.9
        self.spawn_projectile(start, direction, weapon, owner="player")

    def spawn_projectile(self, start, direction, weapon, owner="player", tank_shell=False, ammo_type="HE"):
        vel = Vec3(direction)
        vel.normalize()
        vel *= weapon.muzzle_velocity
        if weapon.name == "RPG":
            node = self.create_rocket_model(start, direction)
        elif tank_shell and ammo_type == "AP":
            node = self.create_ap_dart(start, direction)
        else:
            node = make_sphere(self, self.render, "projectile", start, weapon.projectile_scale, weapon.color, make_material(weapon.color, (1, 0.85, 0.3, 1), 80))
        bullet = BulletProjectile(start, vel, weapon, node, owner, tank_shell, ammo_type)
        bullet.source_pos = Point3(start)
        if owner == "enemy":
            self.enemy_projectiles.append(bullet)
        else:
            self.projectiles.append(bullet)

    def create_rocket_model(self, start, direction):
        root = self.render.attachNewNode("rpg-rocket")
        root.setPos(start)
        root.lookAt(start + direction)
        make_cylinder_approx(root, "body", 0.11, 0.82, (0.12, 0.13, 0.10, 1), (0, 0, 0), (90, 0, 0), 16)
        make_box(root, "warhead", (0.28, 0.28, 0.26), (0.42, 0.42, 0.32, 1), (0, -0.48, 0))
        for a in [0, 90, 180, 270]:
            fin = make_box(root, "fin", (0.04, 0.20, 0.16), (0.08, 0.08, 0.07, 1), (0, 0.45, 0), (0, 0, a))
        return root

    def create_ap_dart(self, start, direction):
        root = self.render.attachNewNode("ap-dart")
        root.setPos(start)
        root.lookAt(start + direction)
        make_cylinder_approx(root, "dart", 0.045, 1.2, (0.55, 0.55, 0.50, 1), (0, 0, 0), (90, 0, 0), 12)
        make_box(root, "tip", (0.14, 0.18, 0.14), (0.75, 0.75, 0.65, 1), (0, -0.65, 0))
        return root

    def start_grenade_charge(self):
        if self.in_apc or self.in_tank or self.grenades <= 0 or self.paused or self.grenade_charging:
            return
        self.grenade_charging = True
        self.grenade_charge = 0.0
        self.show_message("Holding grenade... release G to throw", 0.6)

    def release_grenade(self):
        if not self.grenade_charging:
            return
        charge = clamp(self.grenade_charge, 0.0, 1.45)
        self.grenade_charging = False
        self.throw_grenade(power=charge)

    def throw_grenade(self, power=0.0):
        if self.in_apc or self.in_tank or self.grenades <= 0 or self.paused:
            return
        self.grenades -= 1
        direction = self.camera.getQuat(self.render).getForward()
        direction.normalize()
        start = self.player_pos + direction * 0.9
        # Hold G for longer = higher throw impulse. Fuse is always 4 seconds.
        strength = 10.0 + 13.0 * clamp(power / 1.45, 0.0, 1.0)
        up_boost = 3.2 + 4.8 * clamp(power / 1.45, 0.0, 1.0)
        vel = Vec3(direction) * strength + Vec3(0, 0, up_boost)
        node = make_sphere(self, self.render, "grenade", start, 0.13, (0.08, 0.16, 0.08, 1))
        self.grenade_objects.append(Grenade(start, vel, node, fuse_time=4.0))

    def update_projectiles(self, dt):
        self.update_projectile_list(self.projectiles, dt)
        self.update_projectile_list(self.enemy_projectiles, dt)

    def update_projectile_list(self, plist, dt):
        remaining = []
        for b in plist:
            old = Point3(b.pos)
            b.vel.z -= 9.81 * b.weapon.gravity_scale * dt
            b.pos += b.vel * dt
            b.ttl -= dt
            b.node.setPos(b.pos)
            try:
                b.node.lookAt(b.pos + b.vel)
            except Exception:
                pass
            hit = False
            # terrain
            if b.pos.z <= self.terrain_height(b.pos.x, b.pos.y):
                hit = True
                if b.weapon.splash_radius > 0:
                    self.spawn_explosion(b.pos, b.weapon.splash_radius, b.weapon.splash_damage, source=b.weapon.name)
                else:
                    self.spawn_dust(b.pos, radius=0.7 if b.ammo_type == "AP" else 0.35)
            # obstacles
            if not hit:
                for ox, oy, sx, sy, sz in self.obstacles:
                    if abs(b.pos.x - ox) < sx / 2 and abs(b.pos.y - oy) < sy / 2 and b.pos.z < self.terrain_height(ox, oy) + sz + 0.2:
                        hit = True
                        if b.weapon.splash_radius > 0:
                            self.spawn_explosion(b.pos, b.weapon.splash_radius, b.weapon.splash_damage, source=b.weapon.name)
                        else:
                            self.spawn_sparks(b.pos, b.vel)
                        break
            # direct hits
            if not hit:
                if b.owner == "player":
                    hit = self.check_player_projectile_hits(old, b)
                else:
                    hit = self.check_enemy_projectile_hits(old, b)
            if hit or b.ttl <= 0:
                b.node.removeNode()
            else:
                remaining.append(b)
        plist[:] = remaining

    def check_player_projectile_hits(self, old_pos, b):
        samples = 5
        for i in range(samples + 1):
            p = old_pos * (1 - i / samples) + b.pos * (i / samples)
            # infantry
            for enemy in list(self.enemies):
                if not enemy.alive or enemy.uid in b.hit_ids:
                    continue
                rel = p - enemy.pos
                horiz = math.sqrt(rel.x * rel.x + rel.y * rel.y)
                if horiz <= enemy.radius and 0 <= rel.z <= enemy.height:
                    zone = "body"
                    if rel.z > enemy.height * 0.76:
                        zone = "head"
                    elif rel.z > enemy.height * 0.42:
                        zone = "chest"
                    kd = Vec3(b.vel)
                    if kd.lengthSquared() > 0:
                        kd.normalize()
                    enemy.apply_damage(b.weapon.damage, kd, b.weapon.knockback, zone)
                    b.hit_ids.add(enemy.uid)
                    if b.weapon.splash_radius > 0:
                        self.spawn_explosion(p, b.weapon.splash_radius, b.weapon.splash_damage, source=b.weapon.name)
                        return True
                    if not b.weapon.overpenetrate_mobs and b.ammo_type != "AP":
                        return True
            # tanks
            for tank in list(self.enemy_tanks):
                if not tank.alive:
                    continue
                if abs(p.x - tank.pos.x) < tank.size.x / 2 and abs(p.y - tank.pos.y) < tank.size.y / 2 and 0 <= p.z - tank.pos.z <= 2.25:
                    tank.damage(b.weapon.damage, b.weapon.name, p, b.ammo_type)
                    if b.weapon.splash_radius > 0:
                        self.spawn_explosion(p, b.weapon.splash_radius, b.weapon.splash_damage, source=b.weapon.name)
                    else:
                        self.spawn_sparks(p, b.vel)
                    return True
            # mob spawners
            for spawner in list(self.spawners):
                if not spawner.alive:
                    continue
                if (spawner.pos + Vec3(0, 0, 0.8) - p).length() < spawner.radius + 0.5:
                    spawner.damage(b.weapon.damage * (2.2 if b.weapon.splash_radius > 0 else 1.0))
                    if b.weapon.splash_radius > 0:
                        self.spawn_explosion(p, b.weapon.splash_radius, b.weapon.splash_damage, source=b.weapon.name)
                    else:
                        self.spawn_sparks(p, b.vel)
                    return True
            # destructibles
            for obj in self.destructibles + self.explosive_barrels:
                if obj.destroyed:
                    continue
                if (obj.pos - p).length() < obj.radius + 0.6:
                    obj.damage(b.weapon.damage * (2.5 if b.weapon.splash_radius > 0 else 1.0), p)
                    if obj.kind == "barrel":
                        obj.destroyed = True
                        obj.node.removeNode()
                        self.spawn_explosion(obj.pos + Vec3(0, 0, 0.6), 6.5, 110, source="barrel")
                    if b.weapon.splash_radius > 0:
                        self.spawn_explosion(p, b.weapon.splash_radius, b.weapon.splash_damage, source=b.weapon.name)
                    return True
        return False

    def check_enemy_projectile_hits(self, old_pos, b):
        samples = 4
        target = self.get_combat_target_pos()
        for i in range(samples + 1):
            p = old_pos * (1 - i / samples) + b.pos * (i / samples)
            if (p - target).length() < 1.0:
                if b.weapon.splash_radius > 0:
                    self.spawn_explosion(p, b.weapon.splash_radius, b.weapon.splash_damage, source="enemy")
                else:
                    self.damage_player(b.weapon.damage, getattr(b, "source_pos", p))
                    self.spawn_sparks(p, b.vel)
                return True
        return False

    # ------------------------------------------------------------
    # Explosions, dust, grenades
    # ------------------------------------------------------------

    def create_half_dome_mesh(self, name, radius=1.0, height_scale=0.85, rings=8, segments=28, color=(1, 0.42, 0.06, 0.45)):
        fmt = GeomVertexFormat.getV3n3c4()
        vdata = GeomVertexData(name, fmt, Geom.UHDynamic)
        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        colour = GeomVertexWriter(vdata, "color")
        for r in range(rings + 1):
            phi = (math.pi / 2) * r / rings
            rr = math.sin(phi) * radius
            z = math.cos(phi) * radius * height_scale
            for s in range(segments + 1):
                a = math.tau * s / segments
                x = math.cos(a) * rr
                y = math.sin(a) * rr
                n = Vec3(x, y, z)
                if n.lengthSquared() > 0:
                    n.normalize()
                vertex.addData3(x, y, z)
                normal.addData3(n)
                colour.addData4(*color)
        tris = GeomTriangles(Geom.UHStatic)
        cols = segments + 1
        for r in range(rings):
            for s in range(segments):
                i = r * cols + s
                tris.addVertices(i, i + 1, i + cols)
                tris.addVertices(i + 1, i + cols + 1, i + cols)
        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode(name)
        node.addGeom(geom)
        return node

    def spawn_explosion(self, pos, radius=5.0, damage=80, source="explosion"):
        root = self.render.attachNewNode("explosion")
        root.setPos(pos)

        dome_core = root.attachNewNode(self.create_half_dome_mesh("explosion-dome-core", 1.0, 0.96, 12, 38))
        dome_core.setTransparency(TransparencyAttrib.MAlpha)
        dome_core.setLightOff()
        dome_shell = root.attachNewNode(self.create_half_dome_mesh("explosion-dome-shell", 1.0, 0.92, 10, 34))
        dome_shell.setTransparency(TransparencyAttrib.MAlpha)
        dome_shell.setLightOff()
        flash = make_sphere(self, root, "flash", (0, 0, 0.65), 0.42, (1.0, 0.68, 0.22, 0.92))
        flash.setTransparency(TransparencyAttrib.MAlpha)
        dust_ring = root.attachNewNode(self.create_half_dome_mesh("explosion-dust-ring", 1.0, 0.20, 6, 28))
        dust_ring.setTransparency(TransparencyAttrib.MAlpha)
        dust_ring.setLightOff()
        dust_ring.setZ(0.05)

        sparks = []
        for _ in range(34):
            sp = make_sphere(self, root, "spark", (0, 0, 0.45), random.uniform(0.025, 0.070), (1.0, random.uniform(0.35, 0.95), 0.06, 1))
            d = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.15, 1.45))
            if d.lengthSquared() > 0:
                d.normalize()
            sparks.append({"node": sp, "vel": d * random.uniform(5, 16), "life": random.uniform(0.35, 1.05), "age": 0})

        smoke = []
        for _ in range(18):
            puff = make_sphere(self, root, "smoke", (0, 0, 0.35), random.uniform(0.16, 0.42), (0.13, 0.12, 0.11, 0.55))
            puff.setTransparency(TransparencyAttrib.MAlpha)
            d = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.2, 1.0))
            if d.lengthSquared() > 0:
                d.normalize()
            smoke.append({"node": puff, "vel": d * random.uniform(1.2, 4.4), "age": 0, "life": random.uniform(1.0, 1.8), "base": puff.getScale().x})

        embers = []
        for _ in range(18):
            ember = make_sphere(self, root, "ember", (0, 0, 0.55), random.uniform(0.03, 0.055), (1.0, 0.32, 0.08, 0.92))
            ember.setTransparency(TransparencyAttrib.MAlpha)
            d = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.4, 1.2))
            if d.lengthSquared() > 0:
                d.normalize()
            embers.append({"node": ember, "vel": d * random.uniform(2.0, 7.0), "age": 0, "life": random.uniform(0.5, 1.2)})

        light = PointLight("explosion-light")
        light.setColor((1.0, 0.52, 0.16, 1))
        light_np = root.attachNewNode(light)
        light_np.setPos(0, 0, 2.4)
        self.render.setLight(light_np)

        self.active_explosions.append({
            "root": root,
            "dome_core": dome_core,
            "dome_shell": dome_shell,
            "dust_ring": dust_ring,
            "flash": flash,
            "sparks": sparks,
            "smoke": smoke,
            "embers": embers,
            "light": light_np,
            "age": 0.0,
            "life": 1.75,
            "radius": radius,
            "source": source,
        })
        self.apply_explosion_damage(pos, radius, damage, source)

    def apply_explosion_damage(self, center, radius, damage, source):
        for enemy in list(self.enemies):
            if not enemy.alive:
                continue
            ep = enemy.pos + Vec3(0, 0, 0.9)
            dist = (ep - center).length()
            if dist <= radius:
                falloff = max(0.15, 1.0 - dist / radius)
                direction = ep - center
                enemy.apply_damage(damage * falloff, direction, 8.0 * falloff, "body")
        for tank in list(self.enemy_tanks):
            if not tank.alive:
                continue
            dist = (tank.pos + Vec3(0, 0, 1) - center).length()
            if dist <= radius:
                falloff = max(0.12, 1.0 - dist / radius)
                tank.damage(damage * 0.95 * falloff, source, center, "HE")
        for spawner in list(self.spawners):
            if not spawner.alive:
                continue
            dist = (spawner.pos - center).length()
            if dist <= radius:
                spawner.damage(damage * max(0.15, 1.0 - dist / radius))
        for obj in self.destructibles + self.explosive_barrels:
            if obj.destroyed:
                continue
            dist = (obj.pos - center).length()
            if dist <= radius:
                obj.damage(damage * (1.0 - dist / radius), center)

        target = self.get_combat_target_pos()
        player_dist = (target - center).length()
        if player_dist <= radius:
            falloff = max(0.15, 1.0 - player_dist / radius)
            if source == "mortar":
                player_mult = 1.0
            elif source in ("tank_HE", "grenade", "explosion"):
                player_mult = 0.48
            else:
                player_mult = 0.35
            self.damage_player(damage * player_mult * falloff, center)

    def update_explosions(self, dt):
        remaining = []
        for e in self.active_explosions:
            e["age"] += dt
            t = clamp(e["age"] / e["life"], 0, 1)
            core_scale = e["radius"] * (0.15 + 0.95 * (1 - (1 - t) ** 3))
            shell_scale = e["radius"] * (0.24 + 1.18 * t)
            e["dome_core"].setScale(core_scale, core_scale, core_scale * 0.96)
            e["dome_shell"].setScale(shell_scale, shell_scale, shell_scale * 1.02)
            e["dome_core"].setColor(1.0, 0.52 - 0.12 * t, 0.10, max(0.0, 0.68 * (1 - t)))
            e["dome_shell"].setColor(0.34, 0.32, 0.30, max(0.0, 0.30 * (1 - t) + 0.10))
            ring_scale = e["radius"] * (0.35 + 1.45 * t)
            e["dust_ring"].setScale(ring_scale, ring_scale, max(0.05, 0.16 * (1 - t)))
            e["dust_ring"].setColor(0.42, 0.35, 0.26, max(0.0, 0.42 * (1 - t)))
            e["flash"].setScale(max(0.01, 0.95 * (1 - t)))
            e["flash"].setColor(1.0, 0.66, 0.18, max(0.0, 0.96 * (1 - t)))
            if e["light"] and not e["light"].isEmpty():
                e["light"].node().setColor((1.0 * (1 - t), 0.52 * (1 - t), 0.14 * (1 - t), 1))

            alive_sparks = []
            for s in e["sparks"]:
                s["age"] += dt
                if s["age"] >= s["life"]:
                    s["node"].removeNode()
                    continue
                s["vel"] += Vec3(0, 0, -9.81) * dt * 0.32
                s["node"].setPos(s["node"].getPos() + s["vel"] * dt)
                alive_sparks.append(s)
            e["sparks"] = alive_sparks

            alive_smoke = []
            for sm in e["smoke"]:
                sm["age"] += dt
                if sm["age"] >= sm["life"]:
                    sm["node"].removeNode()
                    continue
                sm_t = sm["age"] / sm["life"]
                sm["node"].setPos(sm["node"].getPos() + sm["vel"] * dt)
                sm["vel"] *= max(0, 1 - dt * 0.65)
                sm["node"].setScale(sm["base"] * (1 + sm_t * 2.7))
                sm["node"].setColor(0.14, 0.13, 0.12, max(0, 0.60 * (1 - sm_t)))
                alive_smoke.append(sm)
            e["smoke"] = alive_smoke

            alive_embers = []
            for em in e["embers"]:
                em["age"] += dt
                if em["age"] >= em["life"]:
                    em["node"].removeNode()
                    continue
                em["vel"] += Vec3(0, 0, -9.81) * dt * 0.12
                em["node"].setPos(em["node"].getPos() + em["vel"] * dt)
                em["node"].setColor(1.0, 0.28, 0.06, max(0.0, 0.92 * (1 - em["age"] / em["life"])))
                alive_embers.append(em)
            e["embers"] = alive_embers

            if t >= 1 and not alive_sparks and not alive_smoke and not alive_embers:
                if e["light"] and not e["light"].isEmpty():
                    self.render.clearLight(e["light"])
                e["root"].removeNode()
            else:
                remaining.append(e)
        self.active_explosions = remaining

    def spawn_dust(self, pos, radius=0.6):
        root = self.render.attachNewNode("dust")
        root.setPos(pos)
        puffs = []
        for _ in range(5):
            puff = make_sphere(self, root, "dust-puff", (0, 0, 0.1), radius * random.uniform(0.10, 0.22), (0.45, 0.38, 0.28, 0.45))
            puff.setTransparency(TransparencyAttrib.MAlpha)
            d = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.15, 0.65))
            if d.lengthSquared() > 0:
                d.normalize()
            puffs.append({"node": puff, "vel": d * random.uniform(0.4, 1.8), "age": 0, "life": 0.75, "base": puff.getScale().x})
        self.active_dust.append({"root": root, "puffs": puffs, "age": 0, "life": 0.8})

    def spawn_sparks(self, pos, vel):
        self.spawn_dust(pos, 0.35)

    def update_dust(self, dt):
        remaining = []
        for d in self.active_dust:
            d["age"] += dt
            p2 = []
            for p in d["puffs"]:
                p["age"] += dt
                if p["age"] >= p["life"]:
                    p["node"].removeNode()
                    continue
                t = p["age"] / p["life"]
                p["node"].setPos(p["node"].getPos() + p["vel"] * dt)
                p["node"].setScale(p["base"] * (1 + t * 2.2))
                p["node"].setColor(0.45, 0.38, 0.28, max(0, 0.45 * (1 - t)))
                p2.append(p)
            d["puffs"] = p2
            if d["age"] >= d["life"] and not p2:
                d["root"].removeNode()
            else:
                remaining.append(d)
        self.active_dust = remaining

    def update_grenades(self, dt):
        remaining = []
        for g in self.grenade_objects:
            g.timer -= dt
            old_pos = Point3(g.pos)
            g.vel.z -= 9.81 * dt
            g.pos += g.vel * dt

            # Bounce off vertical walls/fences/building walls before ground resolution.
            # This is deliberately simple AABB collision, matching the rest of the custom physics.
            for ox, oy, sx, sy, sz in self.obstacles:
                top_z = self.terrain_height(ox, oy) + sz + 0.35
                if g.pos.z > top_z:
                    continue
                inside_now = abs(g.pos.x - ox) < sx / 2 + 0.16 and abs(g.pos.y - oy) < sy / 2 + 0.16
                if not inside_now:
                    continue
                # Decide which axis was crossed and reflect that component.
                was_inside_x = abs(old_pos.x - ox) < sx / 2 + 0.16
                was_inside_y = abs(old_pos.y - oy) < sy / 2 + 0.16
                if not was_inside_x and was_inside_y:
                    g.vel.x *= -0.62
                elif was_inside_x and not was_inside_y:
                    g.vel.y *= -0.62
                else:
                    # If it clipped a corner, reflect the axis with less penetration.
                    pen_x = sx / 2 + 0.16 - abs(g.pos.x - ox)
                    pen_y = sy / 2 + 0.16 - abs(g.pos.y - oy)
                    if pen_x < pen_y:
                        g.vel.x *= -0.62
                    else:
                        g.vel.y *= -0.62
                g.pos = Point3(old_pos)
                g.vel.z *= 0.86
                break

            # Bounce/roll on ground while the 4-second fuse continues burning.
            ground = self.terrain_height(g.pos.x, g.pos.y) + 0.15
            if g.pos.z < ground:
                g.pos.z = ground
                if g.bounces < 3:
                    g.vel.z = abs(g.vel.z) * 0.42
                    g.vel.x *= 0.72
                    g.vel.y *= 0.72
                    g.bounces += 1
                else:
                    g.vel.x *= 0.82
                    g.vel.y *= 0.82
                    if abs(g.vel.z) < 0.7:
                        g.vel.z = 0

            g.node.setPos(g.pos)
            if g.timer <= 0:
                g.node.removeNode()
                self.spawn_explosion(g.pos, 5.3, 105, source="grenade")
            else:
                remaining.append(g)
        self.grenade_objects = remaining

    # ------------------------------------------------------------
    # Enemy fire / damage
    # ------------------------------------------------------------

    def enemy_fire(self, enemy, damage=8):
        target = self.get_combat_target_pos()
        start = enemy.pos + Vec3(0, 0, 1.25)
        if not self.has_line_of_sight(start, target):
            return
        direction = target - start
        if direction.lengthSquared() <= 0:
            return
        direction.normalize()
        w = Weapon("Enemy Rifle", damage, 70, 60, 1, 0, 0, 1.5, 0.12, projectile_scale=0.030, color=(1.0, 0.25, 0.12, 1))
        self.spawn_projectile(start, direction, w, owner="enemy")

    def enemy_tank_fire(self, tank):
        target = self.get_combat_target_pos()
        start = tank.pos + Vec3(0, 0, 1.75)
        if not self.has_line_of_sight(start, target):
            return
        direction = target - start
        if direction.lengthSquared() <= 0:
            return
        direction.normalize()
        w = Weapon("Enemy Tank Shell", 95, 72, 20, 1, 0, 0, 0.5, 0.04, splash_radius=5.0, splash_damage=105, knockback=6.0, projectile_scale=0.10, color=(1.0, 0.30, 0.08, 1))
        self.spawn_projectile(start, direction, w, owner="enemy", tank_shell=True, ammo_type="HE")

    def add_damage_indicator(self, source_pos, amount=1.0):
        """Add a short-lived HUD indicator showing where incoming fire came from."""
        if source_pos is None:
            return
        try:
            src = Point3(source_pos)
        except Exception:
            return
        strength = clamp(float(amount) / 35.0, 0.35, 1.0)
        self.damage_indicators.append({"source": src, "age": 0.0, "life": 1.25, "strength": strength})
        # Keep the list bounded if the player is under heavy fire.
        if len(self.damage_indicators) > 8:
            self.damage_indicators = self.damage_indicators[-8:]

    def damage_player(self, amount, source_pos=None):
        if source_pos is not None and amount > 0:
            self.add_damage_indicator(source_pos, amount)
        if self.in_tank and self.friendly_tank:
            self.friendly_tank.health -= amount
            if self.friendly_tank.health <= 0:
                self.health = 0
                self.show_message("Tank destroyed - you died", 5)
            return
        if self.in_apc and self.apc:
            self.apc.health -= amount
            if self.apc.health <= 0:
                self.health = 0
                self.show_message("APC destroyed - you died", 5)
            return
        self.health -= amount
        if self.health < 0:
            self.health = 0

    # ------------------------------------------------------------
    # Objectives and progression
    # ------------------------------------------------------------

    def objectives_complete(self):
        return self.rescued_hostages >= self.level.required_rescues and self.destroyed_objectives >= self.level.required_destroy

    def complete_level(self):
        if self.level_complete_timer > 0:
            return
        self.level_complete_timer = 3.0
        self.show_message("LEVEL COMPLETE", 3.0)
        self.money += 150
        for i, w in enumerate(self.weapons):
            self.reserves[i] += w.mag_size

    def next_level_or_win(self):
        if self.level_index + 1 < len(self.levels):
            self.load_level(self.level_index + 1)
        else:
            self.paused = True
            self.hud_pause.setText("CAMPAIGN COMPLETE\nQ Quit   R Restart Campaign")
            self.release_mouse()

    def show_message(self, text, duration=2.0):
        self.message_text = text
        self.message_timer = duration

    # ------------------------------------------------------------
    # Drawing HUD/crosshair/minimap/scope
    # ------------------------------------------------------------

    def draw_damage_indicators(self, dt=0.0):
        """
        Red directional hit markers: top = shot from in front, left/right/back accordingly.
        This mimics the common FPS damage-direction indicator without using font glyphs.
        """
        if self.damage_indicator_np:
            self.damage_indicator_np.removeNode()
            self.damage_indicator_np = None

        root = self.aspect2d.attachNewNode("damage-direction-indicators")
        self.damage_indicator_np = root

        remaining = []
        forward = Vec3(-math.sin(math.radians(self.yaw)), math.cos(math.radians(self.yaw)), 0)
        right = Vec3(math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw)), 0)
        radius = 0.48

        for ind in self.damage_indicators:
            ind["age"] += dt
            t = ind["age"] / max(0.001, ind["life"])
            if t >= 1.0:
                continue

            src = ind["source"]
            to_src = Vec3(src.x - self.player_pos.x, src.y - self.player_pos.y, 0)
            if to_src.lengthSquared() <= 0.0001:
                continue
            to_src.normalize()

            sx = to_src.dot(right)
            sz = to_src.dot(forward)
            mag = max(0.001, math.sqrt(sx * sx + sz * sz))
            ux = sx / mag
            uz = sz / mag

            alpha = max(0.0, (1.0 - t)) * (0.55 + 0.45 * ind.get("strength", 0.5))
            col = (1.0, 0.08, 0.03, alpha)
            thick = 3 if ind.get("strength", 0.5) > 0.55 else 2

            cx = ux * radius
            cz = uz * radius
            # Tangent and inward vectors on the HUD plane.
            tx, tz = -uz, ux
            ix, iz = -ux, -uz
            width = 0.060 + 0.020 * ind.get("strength", 0.5)
            depth = 0.040

            a = (cx + tx * width, cz + tz * width)
            b = (cx + ix * depth, cz + iz * depth)
            c = (cx - tx * width, cz - tz * width)
            draw_line(root, "hit-chevron", [a, b, c], col, thick)
            # A short outer bar makes the indicator more visible under heavy fire.
            outer_a = (cx + tx * width * 0.75 + ux * 0.035, cz + tz * width * 0.75 + uz * 0.035)
            outer_b = (cx - tx * width * 0.75 + ux * 0.035, cz - tz * width * 0.75 + uz * 0.035)
            draw_line(root, "hit-outer", [outer_a, outer_b], (1.0, 0.16, 0.08, alpha * 0.75), thick)
            remaining.append(ind)

        self.damage_indicators = remaining

    def draw_crosshair(self):
        if self.crosshair_np:
            self.crosshair_np.removeNode()
        root = self.aspect2d.attachNewNode("crosshair")
        self.crosshair_np = root
        moving = any([self.keys["w"], self.keys["a"], self.keys["s"], self.keys["d"]])
        gap = 0.014 if self.scoped else (0.024 if moving else 0.018)
        length = 0.060 if not self.scoped else 0.043
        col = (0.90, 0.98, 1.0, 0.98)
        # Clean line reticle: no font glyphs, so no missing-character warnings.
        draw_line(root, "ch-l", [(-gap - length, 0), (-gap, 0)], col, 2)
        draw_line(root, "ch-r", [(gap, 0), (gap + length, 0)], col, 2)
        draw_line(root, "ch-u", [(0, gap), (0, gap + length)], col, 2)
        draw_line(root, "ch-d", [(0, -gap), (0, -gap - length)], col, 2)
        # Small centre square/dot drawn from lines.
        d = 0.004
        draw_line(root, "dot", [(-d, -d), (d, -d), (d, d), (-d, d), (-d, -d)], (1.0, 1.0, 1.0, 0.95), 2)
        # Thin outer aiming circle.
        seg = LineSegs("reticle-circle")
        seg.setThickness(1)
        seg.setColor(0.55, 0.82, 1.0, 0.55)
        r = 0.075 if not self.scoped else 0.055
        for i in range(49):
            a = math.tau * i / 48
            x = math.cos(a) * r
            z = math.sin(a) * r
            if i == 0:
                seg.moveTo(x, 0, z)
            else:
                seg.drawTo(x, 0, z)
        root.attachNewNode(seg.create())

    def draw_scope_overlay(self):
        """
        Draw a real scope mask: the world is hidden outside the optic circle.
        This avoids font glyphs and makes scoped view feel like an actual limited-FOV optic.
        HUD/minimap are drawn after this node each frame, so they stay readable.
        """
        if self.scope_np:
            self.scope_np.removeNode()
            self.scope_np = None
        if not self.scoped:
            return

        root = self.aspect2d.attachNewNode("scope-mask-limited-view")
        self.scope_np = root

        # Opaque black annulus with a circular hole in the middle.
        # aspect2d coordinates are aspect-correct, so this appears circular on screen.
        inner_r = 0.56
        outer_r = 2.25
        segs = 144

        fmt = GeomVertexFormat.getV3c4()
        vdata = GeomVertexData("scope-black-annulus", fmt, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, "vertex")
        colour = GeomVertexWriter(vdata, "color")

        for i in range(segs):
            a = math.tau * i / segs
            ca = math.cos(a)
            sa = math.sin(a)
            # outer ring vertex
            vertex.addData3(ca * outer_r, 0, sa * outer_r)
            colour.addData4(0, 0, 0, 1)
            # inner ring vertex
            vertex.addData3(ca * inner_r, 0, sa * inner_r)
            colour.addData4(0, 0, 0, 1)

        tris = GeomTriangles(Geom.UHStatic)
        for i in range(segs):
            ni = (i + 1) % segs
            outer_i = i * 2
            inner_i = i * 2 + 1
            outer_n = ni * 2
            inner_n = ni * 2 + 1
            tris.addVertices(outer_i, outer_n, inner_i)
            tris.addVertices(outer_n, inner_n, inner_i)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("scope-black-annulus")
        node.addGeom(geom)
        root.attachNewNode(node)

        # Thick black rim around the glass.
        rim = LineSegs("scope-heavy-rim")
        rim.setThickness(8)
        rim.setColor(0, 0, 0, 1)
        for i in range(segs + 1):
            a = math.tau * i / segs
            x = math.cos(a) * inner_r
            z = math.sin(a) * inner_r
            if i == 0:
                rim.moveTo(x, 0, z)
            else:
                rim.drawTo(x, 0, z)
        root.attachNewNode(rim.create())

        # Thin glass/reticle ring just inside the rim.
        glass = LineSegs("scope-inner-glass-ring")
        glass.setThickness(2)
        glass.setColor(0.55, 0.85, 1.0, 0.35)
        for i in range(segs + 1):
            a = math.tau * i / segs
            x = math.cos(a) * (inner_r - 0.018)
            z = math.sin(a) * (inner_r - 0.018)
            if i == 0:
                glass.moveTo(x, 0, z)
            else:
                glass.drawTo(x, 0, z)
        root.attachNewNode(glass.create())

        # Scope crosshair lines inside the visible circle.
        ret = LineSegs("scope-reticle-lines")
        ret.setThickness(2)
        ret.setColor(0.02, 0.02, 0.02, 0.90)
        ret.moveTo(-inner_r + 0.07, 0, 0)
        ret.drawTo(-0.055, 0, 0)
        ret.moveTo(0.055, 0, 0)
        ret.drawTo(inner_r - 0.07, 0, 0)
        ret.moveTo(0, 0, -inner_r + 0.07)
        ret.drawTo(0, 0, -0.055)
        ret.moveTo(0, 0, 0.055)
        ret.drawTo(0, 0, inner_r - 0.07)
        # small vertical range marks
        for dz in [-0.24, -0.16, -0.08, 0.08, 0.16, 0.24]:
            ret.moveTo(-0.018, 0, dz)
            ret.drawTo(0.018, 0, dz)
        root.attachNewNode(ret.create())

    def draw_reload_ring(self):
        if self.reload_ring_np:
            self.reload_ring_np.removeNode()
            self.reload_ring_np = None
        if not self.reloading:
            return
        progress = 1.0 - max(0.0, self.reload_timer / max(0.001, self.reload_total))
        root = self.aspect2d.attachNewNode("reload-ring")
        self.reload_ring_np = root
        radius = 0.075
        segs = 54
        red = LineSegs("reload-red")
        red.setThickness(3)
        red.setColor(1.0, 0.05, 0.03, 0.9)
        for i in range(segs + 1):
            a = math.radians(90 - 360 * i / segs)
            x = math.cos(a) * radius
            z = math.sin(a) * radius
            if i == 0:
                red.moveTo(x, 0, z)
            else:
                red.drawTo(x, 0, z)
        root.attachNewNode(red.create())
        green = LineSegs("reload-green")
        green.setThickness(5)
        green.setColor(0.1, 1.0, 0.2, 0.95)
        n = max(2, int(segs * progress))
        for i in range(n + 1):
            a = math.radians(90 - 360 * progress * i / n)
            x = math.cos(a) * radius
            z = math.sin(a) * radius
            if i == 0:
                green.moveTo(x, 0, z)
            else:
                green.drawTo(x, 0, z)
        root.attachNewNode(green.create())

    def draw_minimap(self):
        if self.minimap_np:
            self.minimap_np.removeNode()
        root = self.aspect2d.attachNewNode("minimap")
        self.minimap_np = root
        # Bottom-right radar/minimap.
        cx, cz, radius = 1.05, -0.58, 0.22
        world_range = 55.0

        def wr(wx, wy):
            dx = wx - self.player_pos.x
            dy = wy - self.player_pos.y
            d = math.sqrt(dx * dx + dy * dy)
            if d > world_range:
                return None
            x = cx + dx / world_range * radius
            z = cz + dy / world_range * radius
            if (x - cx) ** 2 + (z - cz) ** 2 > radius ** 2:
                return None
            return x, z

        def circle(name, x, z, r, color, thick=1, segs=48):
            seg = LineSegs(name)
            seg.setThickness(thick)
            seg.setColor(*color)
            for i in range(segs + 1):
                a = math.tau * i / segs
                px, pz = x + math.cos(a) * r, z + math.sin(a) * r
                if i == 0:
                    seg.moveTo(px, 0, pz)
                else:
                    seg.drawTo(px, 0, pz)
            root.attachNewNode(seg.create())

        circle("outer", cx, cz, radius, (0.65, 0.88, 1, 0.95), 2, 72)
        for f in [0.25, 0.5, 0.75]:
            circle("ring", cx, cz, radius * f, (0.35, 0.55, 0.65, 0.7), 1, 48)

        # Buildings, walls, fences, crates and obstacles.
        # Drawn before enemies so the red/blue tactical markers stay readable.
        for ox, oy, sx, sy, sz in self.obstacles:
            p = wr(ox, oy)
            if not p:
                continue
            hw = max(0.002, (sx / world_range) * radius * 0.5)
            hh = max(0.002, (sy / world_range) * radius * 0.5)
            # Larger/taller obstacles are likely building walls; low long ones are fences.
            if sz > 2.4:
                col = (0.72, 0.74, 0.76, 0.78)
                thick = 2
            elif max(sx, sy) > 4.5:
                col = (0.48, 0.52, 0.45, 0.70)
                thick = 1
            else:
                col = (0.58, 0.45, 0.25, 0.62)
                thick = 1
            draw_line(
                root,
                "map-obstacle",
                [(p[0]-hw, p[1]-hh), (p[0]+hw, p[1]-hh), (p[0]+hw, p[1]+hh), (p[0]-hw, p[1]+hh), (p[0]-hw, p[1]-hh)],
                col,
                thick,
            )

        # War-Thunder-like mortar aiming reticle on minimap: target cross + dispersion circle.
        if self.mortar_targeting:
            p = wr(self.mortar_target.x, self.mortar_target.y)
            if p:
                circle("mortar-dispersion", p[0], p[1], radius * 7.8 / world_range, (1.0, 0.18, 0.04, 0.65), 2, 36)
                circle("mortar-aim-inner", p[0], p[1], radius * 3.0 / world_range, (1.0, 0.35, 0.08, 0.55), 1, 24)
                draw_line(root, "mortar-map-cross", [(p[0]-0.018, p[1]), (p[0]+0.018, p[1]), (p[0], p[1]-0.018), (p[0], p[1]+0.018)], (1.0, 0.06, 0.02, 1), 3)

        # exit marker
        p = wr(self.exit_pos.x, self.exit_pos.y)
        if p:
            circle("exit", p[0], p[1], 0.014, (0.05, 1.0, 0.15, 1), 3, 16)
        for sp in self.spawners:
            if sp.alive:
                p = wr(sp.pos.x, sp.pos.y)
                if p:
                    draw_line(root, "spawner", [(p[0]-0.009, p[1]-0.009), (p[0]+0.009, p[1]-0.009), (p[0]+0.009, p[1]+0.009), (p[0]-0.009, p[1]+0.009), (p[0]-0.009, p[1]-0.009)], (1.0, 0.0, 0.5, 1), 2)
        for e in self.enemies:
            if e.alive:
                p = wr(e.pos.x, e.pos.y)
                if p:
                    circle("enemy", p[0], p[1], 0.006, (1, 0.1, 0.05, 1), 2, 12)
        for t in self.enemy_tanks:
            if t.alive:
                p = wr(t.pos.x, t.pos.y)
                if p:
                    draw_line(root, "tank", [(p[0] - 0.010, p[1] - 0.006), (p[0] + 0.010, p[1] - 0.006), (p[0] + 0.010, p[1] + 0.006), (p[0] - 0.010, p[1] + 0.006), (p[0] - 0.010, p[1] - 0.006)], (1.0, 0.25, 0.05, 1), 2)
        if self.apc:
            p = wr(self.apc.pos.x, self.apc.pos.y)
            if p:
                circle("apc", p[0], p[1], 0.011, (0.1, 0.55, 1.0, 1), 2, 14)
        if self.friendly_tank:
            p = wr(self.friendly_tank.pos.x, self.friendly_tank.pos.y)
            if p:
                draw_line(root, "friendly-tank", [(p[0] - 0.012, p[1] - 0.007), (p[0] + 0.012, p[1] - 0.007), (p[0] + 0.012, p[1] + 0.007), (p[0] - 0.012, p[1] + 0.007), (p[0] - 0.012, p[1] - 0.007)], (0.15, 0.7, 1.0, 1), 2)
        # player triangle + V, corrected direction.
        yaw_rad = math.radians(-self.yaw)
        fx, fz = math.sin(yaw_rad), math.cos(yaw_rad)
        rx, rz = math.cos(yaw_rad), -math.sin(yaw_rad)
        tip = (cx + fx * 0.022, cz + fz * 0.022)
        left = (cx - fx * 0.013 - rx * 0.012, cz - fz * 0.013 - rz * 0.012)
        right = (cx - fx * 0.013 + rx * 0.012, cz - fz * 0.013 + rz * 0.012)
        draw_line(root, "tri", [tip, left, right, tip], (0.05, 1, 0.2, 1), 2)
        # Mirror the real field of view on the minimap.
        # Scoped view uses a much narrower cone matching the scope FOV.
        cone = math.radians((self.scope_fov * 0.5) if self.scoped else 25)
        cone_len = radius * (0.88 if self.scoped else 0.72)
        cone_col = (0.2, 1.0, 0.2, 0.95) if self.scoped else (0.1, 1.0, 0.25, 0.8)
        lpt = (cx + math.sin(yaw_rad - cone) * cone_len, cz + math.cos(yaw_rad - cone) * cone_len)
        rpt = (cx + math.sin(yaw_rad + cone) * cone_len, cz + math.cos(yaw_rad + cone) * cone_len)
        draw_line(root, "v1", [(cx, cz), lpt], cone_col, 2 if self.scoped else 1)
        draw_line(root, "v2", [(cx, cz), rpt], cone_col, 2 if self.scoped else 1)
        if self.scoped:
            # Add small crossbar to make the narrowed scoped search cone very obvious.
            draw_line(root, "scope-fov-cap", [lpt, rpt], (0.2, 1.0, 0.2, 0.45), 1)
        OnscreenText("RADAR" + ("  SCOPE FOV" if self.scoped else ""), parent=root, pos=(cx, cz + radius + 0.025), scale=0.025, align=TextNode.ACenter, fg=(0.65, 0.9, 1, 1))

    def update_hud(self):
        w = self.weapons[self.current_weapon]
        if self.in_tank:
            mode = "TANK"
            hp_line = f"TANK HP: {int(self.friendly_tank.health)}/{self.friendly_tank.max_health}"
            reload_txt = f" reload {self.friendly_tank.reload_timer:.1f}s" if self.friendly_tank.reload_timer > 0 else " READY"
            ammo_line = f"Tank {self.friendly_tank.ammo_type}: LMB fire | Z switch |{reload_txt}"
        elif self.in_apc:
            mode = "APC"
            hp_line = f"APC HP: {int(self.apc.health)}/{self.apc.max_health}"
            ammo_line = f"APC LMG: {self.apc.ammo}/{self.apc.reserve}"
        else:
            mode = "ON FOOT"
            hp_line = f"HEALTH: {int(self.health)}/100"
            charge_txt = f"  G charge: {int(100 * self.grenade_charge / 1.45)}%" if self.grenade_charging else ""
            ammo_line = f"{w.name}: {self.magazines[self.current_weapon]}/{self.reserves[self.current_weapon]}   Grenades: {self.grenades}{charge_txt}"
        self.hud_main.setText(
            f"{hp_line}\n"
            f"MODE: {mode}\n"
            f"MONEY: ${self.money}\n"
            f"{ammo_line}\n"
            f"Mortar: ${self.mortar_cost}  {'ARTY SIGHT' if self.mortar_targeting else ('READY' if self.mortar_cooldown <= 0 else str(int(self.mortar_cooldown)) + 's')} | X use\n"
            f"Shift scope | W/Space ladder climb | Z tank HE/AP | Gravity: ON | Kills: {self.kills}"
        )
        self.hud_top.setText(f"{self.level.name}\nBuild: V29 realistic graphics + self-damage mortar")
        obj = (
            f"Objective: {self.level.objective_text}\n"
            f"Rescued {self.rescued_hostages}/{self.level.required_rescues}   "
            f"Destroyed {self.destroyed_objectives}/{self.level.required_destroy}   "
            f"Extraction: {'WALK INTO GREEN ZONE' if self.objectives_complete() else 'LOCKED'}"
        )
        self.hud_obj.setText(obj)
        if self.message_timer > 0:
            self.hud_message.setText(self.message_text)
        else:
            self.hud_message.setText("")

    # ------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------

    def update(self, task):
        dt = globalClock.getDt()
        dt = min(dt, 0.033)
        if self.paused:
            # Q quits only while paused; R restarts level/campaign.
            if self.mouseWatcherNode.isButtonDown("q"):
                self.userExit()
            if self.mouseWatcherNode.isButtonDown("r"):
                self.paused = False
                self.hud_pause.setText("")
                self.lock_mouse()
                if "CAMPAIGN COMPLETE" in self.hud_pause.getText():
                    self.load_level(0)
                else:
                    self.load_level(self.level_index)
            return Task.cont

        if self.health <= 0:
            self.hud_pause.setText("YOU DIED\nR Restart Level   Q Quit")
            self.paused = True
            self.release_mouse()
            return Task.cont

        self.update_player(dt)

        if self.grenade_charging:
            self.grenade_charge = min(1.45, self.grenade_charge + dt)
        self.fire_timer = max(0.0, self.fire_timer - dt)
        if self.reloading:
            self.reload_timer -= dt
            if self.reload_timer <= 0:
                self.finish_reload()
        if self.mouse_down:
            self.try_fire()
        if self.in_apc and self.apc and self.keys["space"]:
            self.apc.fire_lmg()

        for spawner in list(self.spawners):
            spawner.update(dt)
        self.spawners = [sp for sp in self.spawners if sp.alive]
        for enemy in list(self.enemies):
            enemy.update(dt)
        self.enemies = [e for e in self.enemies if e.alive]
        for tank in list(self.enemy_tanks):
            tank.update(dt)
        self.enemy_tanks = [t for t in self.enemy_tanks if t.alive]
        if self.apc and not self.in_apc:
            self.apc.update(dt)
        if self.friendly_tank and not self.in_tank:
            self.friendly_tank.update(dt)

        self.update_projectiles(dt)
        self.update_grenades(dt)
        self.update_mortar(dt)
        self.update_explosions(dt)
        self.update_dust(dt)

        self.message_timer = max(0.0, self.message_timer - dt)
        if self.level_complete_timer > 0:
            self.level_complete_timer -= dt
            if self.level_complete_timer <= 0:
                self.next_level_or_win()

        # Walk into the bright green extraction zone to complete the level.
        if (self.exit_pos - self.player_pos).length() < 6.0 and self.level_complete_timer <= 0:
            if self.objectives_complete():
                self.complete_level()
            else:
                self.show_message("Extraction locked: finish objectives", 0.25)

        self.draw_crosshair()
        self.draw_damage_indicators(dt)
        self.draw_reload_ring()
        self.draw_minimap()
        self.update_hud()
        return Task.cont


if __name__ == "__main__":
    game = FPSGame()
    game.run()
