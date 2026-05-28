# fps_panda3d_realistic_pillbox_defence.py
#
# Install:
#   pip install panda3d
#
# Run:
#   python fps_panda3d_round_pillbox_radar.py
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
#   1/2/3/4 = switch weapon
#   Space = jump
#   Shift = sprint
#   Esc = quit

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
    sphere.setScale(scale)
    sphere.setColor(*color)
    return sphere



def make_projectile_visual(game, parent, weapon, pos, direction):
    """
    Creates a more realistic projectile visual.
    Normal rounds are small bright bullet cores with a faint tracer line.
    RPG rounds are rocket-shaped: dark body, red nose, fins, and exhaust glow.
    """
    root = parent.attachNewNode("projectile-root")
    root.setPos(pos)

    if weapon.name == "RPG":
        # Rocket points along local +Y. We rotate the root with lookAt() each frame.
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

        self.hp_bar_back = make_box(
            self.root,
            "hp-back",
            size=(0.85 * s, 0.05, 0.05),
            color=(0.04, 0.04, 0.04, 1),
            pos=(0, -0.3, 2.15 * s)
        )

        self.hp_bar = make_box(
            self.root,
            "hp",
            size=(0.85 * s, 0.06, 0.06),
            color=(0.1, 0.85, 0.1, 1),
            pos=(0, -0.34, 2.15 * s)
        )

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

        if dist > attack_distance:
            side = Vec3(direction.y, -direction.x, 0)
            zigzag = math.sin(self.game.time * 2.0 + pos.x * 0.25) * 0.25
            separation = self.separation_force()
            movement = direction * self.speed + side * zigzag + separation

            if movement.length() > self.speed * 1.8:
                movement.normalize()
                movement *= self.speed * 1.8

            step = movement * dt

            new_x = pos.x + step.x
            new_y = pos.y + step.y
            new_z = self.game.terrain_height(new_x, new_y)

            self.root.setPos(new_x, new_y, new_z)

        self.attack_cooldown = max(0.0, self.attack_cooldown - dt)

        if dist <= attack_distance and self.attack_cooldown <= 0.0:
            if self.game.pillbox_health > 0:
                self.game.damage_pillbox(self.damage)
            else:
                self.game.damage_player(self.damage)

            self.attack_cooldown = 0.85

        ratio = max(0.0, self.health / self.max_health)
        self.hp_bar.setScale(ratio, 1, 1)
        self.hp_bar.setX(-(1.0 - ratio) * 0.42)

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
                damage=55,
                muzzle_velocity=55,
                rpm=28,
                mag_size=1,
                reserve=8,
                reload_time=2.7,
                spread_deg=0.15,
                gravity_scale=0.0,
                pellets=1,
                splash_radius=5.8,
                splash_damage=95,
                projectile_scale=0.16,
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

        self.reload_ring_np = None
        self.minimap_np = None
        self.scope_np = None

        self.setup_lighting()
        self.create_terrain()
        self.create_round_pillbox()
        self.create_cover_and_details()
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
        # Flatter, more realistic rolling terrain.
        hills = (
            0.62 * math.sin(x * 0.055) * math.cos(y * 0.045)
            + 0.24 * math.sin((x + y) * 0.085)
            + 0.12 * math.sin(x * 0.15)
        )

        trench_front = -0.65 * math.exp(-((y + 2) ** 2) / 8.5)
        trench_side = -0.45 * math.exp(-((x - 15) ** 2) / 8.5)

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
            if i % 4 == 0:
                continue

            angle = math.tau * i / segments
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
        self.hud_health = OnscreenText(text="", pos=(-1.28, 0.90), scale=0.055, align=TextNode.ALeft, fg=(1, 1, 1, 1), mayChange=True)
        self.hud_pillbox = OnscreenText(text="", pos=(-1.28, 0.82), scale=0.052, align=TextNode.ALeft, fg=(0.95, 0.95, 0.95, 1), mayChange=True)
        self.hud_info = OnscreenText(text="", pos=(-1.28, 0.74), scale=0.04, align=TextNode.ALeft, fg=(0.95, 0.95, 0.95, 1), mayChange=True)
        self.hud_ammo = OnscreenText(text="", pos=(1.27, -0.88), scale=0.058, align=TextNode.ARight, fg=(1, 1, 1, 1), mayChange=True)
        self.hud_weapons = OnscreenText(text="", pos=(0, -0.93), scale=0.044, align=TextNode.ACenter, fg=(0.95, 0.95, 0.95, 1), mayChange=True)
        self.crosshair_np = None
        self.hud_hit = OnscreenText(text="", pos=(0, -0.13), scale=0.07, align=TextNode.ACenter, fg=(1, 0.90, 0.20, 1), mayChange=True)
        self.hud_warning = OnscreenText(text="", pos=(0, 0.82), scale=0.06, align=TextNode.ACenter, fg=(1, 0.25, 0.2, 1), mayChange=True)
        self.scope_label = OnscreenText(text="", pos=(0, 0.72), scale=0.045, align=TextNode.ACenter, fg=(0.75, 1.0, 0.75, 1), mayChange=True)

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

        if self.pillbox_health > 0:
            self.hud_health.setText(f"Player: protected / {self.health}")
        else:
            self.hud_health.setText(f"Player: {max(0, int(self.health))}/100")

        self.hud_pillbox.setText(f"Pillbox: {max(0, int(self.pillbox_health))}/{self.pillbox_max_health}")
        self.hud_info.setText(f"Wave {self.wave}: {self.current_wave_name()}   Kills {self.wave_kills}/{self.wave_goal}   Enemies {len(self.enemies)}")
        self.hud_ammo.setText(f"{w.name}   {mag}/{reserve}")

        slots = []
        for i, weapon in enumerate(self.weapons):
            if i == self.current_weapon:
                slots.append(f"[{i + 1}:{weapon.name}]")
            else:
                slots.append(f" {i + 1}:{weapon.name} ")
        self.hud_weapons.setText("   ".join(slots))

        self.scope_label.setText("SCOPE" if self.scoped else "")

        if self.reloading:
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

        for enemy in self.enemies:
            if not enemy.alive:
                continue
            ep = enemy.world_pos()
            p = world_to_radar(ep.x, ep.y)
            if p is not None:
                ex, ez = p
                add_circle("enemy-o", ex, ez, 0.0075, (1.0, 0.15, 0.08, 1.0), 2, 14)

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
        for key in ["w", "a", "s", "d", "space", "shift"]:
            self.accept(key, self.set_key, [key, True])
            self.accept(key + "-up", self.set_key, [key, False])

        self.accept("mouse1", self.set_mouse, [True])
        self.accept("mouse1-up", self.set_mouse, [False])
        self.accept("r", self.start_reload)
        self.accept("q", self.toggle_scope)
        self.accept("1", self.switch_weapon, [0])
        self.accept("2", self.switch_weapon, [1])
        self.accept("3", self.switch_weapon, [2])
        self.accept("4", self.switch_weapon, [3])
        self.accept("escape", self.userExit)

    def set_key(self, key, value):
        self.keys[key] = value

    def set_mouse(self, value):
        self.mouse_down = value

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

        yaw_rad = math.radians(self.yaw)
        forward = Vec3(math.sin(yaw_rad), math.cos(yaw_rad), 0)
        right = Vec3(math.cos(yaw_rad), -math.sin(yaw_rad), 0)

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

        if dist > max_dist:
            scale = max_dist / max(0.001, dist)
            self.player_pos.x = self.pillbox_pos.x + dx * scale
            self.player_pos.y = self.pillbox_pos.y + dy * scale

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
        if self.game_over or self.reloading:
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

        bullet_node = make_sphere(self, self.render, "bullet", start, weapon.projectile_scale, weapon.projectile_color)
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
            hit = False

            if bullet.pos.z <= self.terrain_height(bullet.pos.x, bullet.pos.y):
                hit = True
                self.create_impact_effect(bullet.pos, bullet.weapon)
                if bullet.weapon.splash_radius > 0:
                    self.apply_splash_damage(bullet.pos, bullet.weapon)

            if not hit:
                hit = self.check_bullet_enemy_hit(old_pos, bullet.pos, bullet)

            if hit or bullet.ttl <= 0:
                bullet.node.removeNode()
            else:
                remaining.append(bullet)

        self.bullets = remaining

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

                enemy.apply_damage(bullet.damage * multiplier)
                self.show_hit_marker(label)

                if bullet.weapon.splash_radius > 0:
                    self.apply_splash_damage(p, bullet.weapon, direct_hit_enemy=enemy)
                    self.create_impact_effect(p, bullet.weapon)

                return True

        return False

    def apply_splash_damage(self, pos, weapon, direct_hit_enemy=None):
        for enemy in list(self.enemies):
            if not enemy.alive:
                continue
            if direct_hit_enemy is not None and enemy is direct_hit_enemy:
                continue
            ep = enemy.world_pos() + Vec3(0, 0, 1.0)
            dist = (ep - pos).length()
            if dist <= weapon.splash_radius:
                falloff = 1.0 - dist / weapon.splash_radius
                damage = weapon.splash_damage * max(0.15, falloff)
                enemy.apply_damage(damage)
                self.show_hit_marker("SPLASH")

    def create_impact_effect(self, pos, weapon):
        if weapon.splash_radius <= 0:
            return

        root = self.render.attachNewNode("rpg-explosion-effect")
        root.setPos(pos)

        # Flash core.
        flash = make_sphere(
            self,
            root,
            "explosion-flash",
            Point3(0, 0, 0.15),
            weapon.splash_radius * 0.16,
            (1.0, 0.55, 0.08, 0.70)
        )
        flash.setTransparency(True)

        # Expanding shock ring on the ground.
        ring = LineSegs("shock-ring")
        ring.setThickness(5)
        ring.setColor(1.0, 0.72, 0.20, 0.85)
        r = weapon.splash_radius * 0.42
        segments = 72
        for i in range(segments + 1):
            a = math.tau * i / segments
            x = math.cos(a) * r
            y = math.sin(a) * r
            if i == 0:
                ring.moveTo(x, y, 0.08)
            else:
                ring.drawTo(x, y, 0.08)
        root.attachNewNode(ring.create())

        # Smoke/debris puffs, non-bloody.
        for i in range(12):
            a = math.tau * i / 12
            dist = random.uniform(0.25, weapon.splash_radius * 0.35)
            puff = make_sphere(
                self,
                root,
                "smoke-puff",
                Point3(math.cos(a) * dist, math.sin(a) * dist, random.uniform(0.10, 0.85)),
                random.uniform(0.10, 0.22),
                (0.30, 0.28, 0.24, 0.62)
            )
            puff.setTransparency(True)

        def remove_effect(task):
            root.removeNode()
            return Task.done

        self.taskMgr.doMethodLater(0.26, remove_effect, "remove-rpg-explosion-effect")

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
            return random.choices(self.enemy_types, weights=[0.45, 0.15, 0.40], k=1)[0]
        if pattern == 2:
            return random.choices(self.enemy_types, weights=[0.75, 0.05, 0.20], k=1)[0]
        if pattern == 3:
            return random.choices(self.enemy_types, weights=[0.20, 0.55, 0.25], k=1)[0]
        return random.choice(self.enemy_types)

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
        self.spawn_timer -= dt
        target_count = min(18, 4 + self.wave * 2)

        if self.spawn_timer <= 0 and len(self.enemies) < target_count and self.wave_kills < self.wave_goal:
            self.spawn_enemy()
            self.spawn_timer = max(0.55, 1.75 - self.wave * 0.06)

        for enemy in list(self.enemies):
            enemy.update(dt)

        self.enemies = [e for e in self.enemies if e.alive]

        if self.wave_kills >= self.wave_goal and len(self.enemies) == 0:
            self.next_wave()

    def next_wave(self):
        self.wave += 1
        self.wave_kills = 0
        self.wave_goal = 10 + self.wave * 3
        self.health = min(self.max_health, self.health + 25)
        self.pillbox_health = min(self.pillbox_max_health, self.pillbox_health + 60)

        for i, w in enumerate(self.weapons):
            if w.name == "RPG":
                self.reserves[i] += 2
            else:
                self.reserves[i] += w.mag_size * 2

    # ----------------------------
    # Damage
    # ----------------------------

    def damage_player(self, amount):
        if self.game_over:
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
        self.time += dt

        if not self.game_over:
            self.update_player(dt)
            self.fire_timer = max(0.0, self.fire_timer - dt)

            if self.reloading:
                self.reload_timer -= dt
                if self.reload_timer <= 0:
                    self.finish_reload()

            if self.mouse_down:
                self.try_fire()

            self.update_bullets(dt)
            self.update_enemies(dt)

        self.hit_marker_timer = max(0.0, self.hit_marker_timer - dt)
        self.damage_flash_timer = max(0.0, self.damage_flash_timer - dt)

        moving = any([self.keys["w"], self.keys["a"], self.keys["s"], self.keys["d"]])

        self.draw_crosshair(moving=moving, firing=self.fire_timer > 0.03)

        self.update_hud()
        self.draw_minimap()
        return Task.cont


if __name__ == "__main__":
    game = FPSGame()
    game.run()
