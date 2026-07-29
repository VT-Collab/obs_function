import copy
import logging
import itertools
from collections import Counter, defaultdict
from time import time
from typing import Dict, List, Tuple
import numpy as np
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState  # PlayerState,
from overcooked_ai_py.mdp.overcooked_mdp import (
    Action,
    Direction,
    ObjectState,
    OvercookedGridworld,
    Recipe,
    SoupState,
)
from overcooked_ai_py.utils import read_layout_dict
import random

logger = logging.getLogger(__name__)

class PlayerState:
    """State of a player in SteakhouseGridworld.

    position: (x, y) tuple representing the player's location.
    orientation: Direction.NORTH/SOUTH/EAST/WEST representing orientation.
    held_object: ObjectState representing the object held by the player, or
        None if there is no such object.
    num_ingre_held (int): Number of times the player has held an ingredient
        object.
    num_plate_held (int): Number of times the player has held a plate
    num_served (int): Number of times the player has served food
    """

    def __init__(
        self,
        position,
        orientation,
        held_object=None,
        num_ingre_held=0,
        num_plate_held=0,
        num_served=0,
        active_log=[],
        stuck_log=[],
        subtask_log=[],
        utter=None,
    ):
        self.position = tuple(position)
        self.orientation = tuple(orientation)
        self.held_object = held_object
        self.num_ingre_held = num_ingre_held
        self.num_plate_held = num_plate_held
        self.num_served = num_served
        self.active_log = active_log.copy()
        self.stuck_log = stuck_log.copy()
        self.subtask_log = subtask_log.copy()
        self.utter = utter

        assert self.orientation in Direction.ALL_DIRECTIONS
        if self.held_object is not None:
            assert isinstance(self.held_object, ObjectState)
            assert self.held_object.position == self.position

    @property
    def pos_and_or(self):
        return self.position, self.orientation

    def get_pos_and_or(self):
        return self.position, self.orientation

    def has_object(self):
        return self.held_object is not None

    def get_object(self):
        assert self.has_object()
        return self.held_object

    def set_object(self, obj):
        assert not self.has_object()
        obj.position = self.position
        self.held_object = obj

    def remove_object(self):
        assert self.has_object()
        obj = self.held_object
        self.held_object = None
        return obj

    def update_pos_and_or(self, new_position, new_orientation):
        self.position = new_position
        self.orientation = new_orientation
        if self.has_object():
            self.get_object().position = new_position

    def deepcopy(self):
        new_obj = None if self.held_object is None else self.held_object.deepcopy()
        return PlayerState(
            self.position,
            self.orientation,
            new_obj,
            self.num_ingre_held,
            self.num_plate_held,
            self.num_served,
            self.active_log,
            self.stuck_log,
            self.subtask_log,
            self.utter,
        )

    def __eq__(self, other):
        return (
            isinstance(other, PlayerState)
            and self.position == other.position
            and self.orientation == other.orientation
            and self.held_object == other.held_object
            and self.utter == other.utter
        )

    def __hash__(self):
        return hash((self.position, self.orientation, self.held_object, self.utter))

    def __repr__(self):
        return (
            f"{self.position} facing {self.orientation} holding "
            f"{str(self.held_object)} with utterance {self.utter}"
        )

    def to_dict(self):
        return {
            "position": self.position,
            "orientation": self.orientation,
            "held_object": (
                self.held_object.to_dict() if self.held_object is not None else None
            ),
            "utter": self.utter,
        }

    def get_workload(
        self,
    ):
        return {
            "num_ingre_held": self.num_ingre_held,
            "num_plate_held": self.num_plate_held,
            "num_served": self.num_served,
        }

    def print_workload(
        self,
    ):
        logger.info(f"Number of ingredients held: {self.num_ingre_held}")
        logger.info(f"Number of plates held: {self.num_plate_held}")
        logger.info(f"Number of soup served: {self.num_served}")

    @staticmethod
    def from_dict(player_dict):
        player_dict = copy.deepcopy(player_dict)
        held_obj = player_dict["held_object"]
        if held_obj is not None:
            player_dict["held_object"] = ObjectState.from_dict(held_obj)
        return PlayerState(**player_dict)


class Steakhouse_Recipe(Recipe):
    MAX_NUM_INGREDIENTS = 2
    CHICKEN = "chicken"
    MEAT = "meat"
    ONION = "onion"

    ALL_INGREDIENTS = [CHICKEN, MEAT, ONION]
    STR_REP = {CHICKEN: "@", MEAT: "!", ONION: "ø"}

    @classmethod
    def configure(cls, conf):
        cls._conf = conf
        cls._configured = True
        cls._computed = False
        cls.MAX_NUM_INGREDIENTS = conf.get("max_num_ingredients", 2)

        cls._cook_time = None
        cls.delivery_reward = None
        cls.in_order_delivery_reward = None
        cls.non_order_delivery_reward = None
        cls._value_mapping = None
        cls._time_mapping = None
        cls._onion_value = None
        cls._steak_time = None
        cls._chicken_value = None
        cls._chicken_time = None

        ## Basic checks for validity ##

        # Mutual Exclusion
        if (
            "chicken_time" in conf
            and not "steak_time" in conf
            or "steak_time" in conf
            and not "chicken_time" in conf
        ):
            raise ValueError("Must specify both 'steak_time' and 'chicken_time'")
        if (
            "chicken_value" in conf
            and not "steak_value" in conf
            or "steak_value" in conf
            and not "chicken_value" in conf
        ):
            raise ValueError("Must specify both 'steak_value' and 'chicken_value'")
        if "chicken_value" in conf and "delivery_reward" in conf:
            raise ValueError("'delivery_reward' incompatible with '<ingredient>_value'")
        if "chicken_value" in conf and "recipe_values" in conf:
            raise ValueError("'recipe_values' incompatible with '<ingredient>_value'")
        if "recipe_values" in conf and "delivery_reward" in conf:
            raise ValueError("'delivery_reward' incompatible with 'recipe_values'")
        if "chicken_time" in conf and "cook_time" in conf:
            raise ValueError("'cook_time' incompatible with '<ingredient>_time")
        if "chicken_time" in conf and "recipe_times" in conf:
            raise ValueError("'recipe_times' incompatible with '<ingredient>_time'")
        if "recipe_times" in conf and "cook_time" in conf:
            raise ValueError("'delivery_reward' incompatible with 'recipe_times'")

        # recipe_ lists and orders compatibility
        if "recipe_values" in conf:
            if not "all_orders" in conf or not conf["all_orders"]:
                raise ValueError(
                    "Must specify 'all_orders' if 'recipe_values' specified"
                )
            if not len(conf["all_orders"]) == len(conf["recipe_values"]):
                raise ValueError(
                    "Number of recipes in 'all_orders' must be the same as number in 'recipe_values"
                )
        if "recipe_times" in conf:
            if not "all_orders" in conf or not conf["all_orders"]:
                raise ValueError(
                    "Must specify 'all_orders' if 'recipe_times' specified"
                )
            if not len(conf["all_orders"]) == len(conf["recipe_times"]):
                raise ValueError(
                    "Number of recipes in 'all_orders' must be the same as number in 'recipe_times"
                )

        ## Conifgure ##

        if "cook_time" in conf:
            cls._cook_time = conf["cook_time"]

        if "delivery_reward" in conf:
            cls.delivery_reward = conf["delivery_reward"]

        if "in_order_delivery_reward" in conf:
            cls.in_order_delivery_reward = conf["in_order_delivery_reward"]

        if "non_order_delivery_reward" in conf:
            cls.non_order_delivery_reward = conf["non_order_delivery_reward"]

        if "recipe_values" in conf:
            cls._value_mapping = {
                cls.from_dict(recipe): value
                for (recipe, value) in zip(conf["all_orders"], conf["recipe_values"])
            }

        if "recipe_times" in conf:
            cls._time_mapping = {
                cls.from_dict(recipe): time
                for (recipe, time) in zip(conf["all_orders"], conf["recipe_times"])
            }

        if "chicken_time" in conf:
            cls._chicken_time = conf["chicken_time"]

        if "steak_time" in conf:
            cls._steak_time = conf["steak_time"]

        if "chicken_value" in conf:
            cls._chicken_value = conf["chicken_value"]

        if "steak_value" in conf:
            cls._steak_value = conf["steak_value"]

        if "wash_time" in conf:
            cls._wash_time = conf["wash_time"]


class IdObjectState(ObjectState):
    def __init__(self, id, name, position):
        self.id = id
        super(IdObjectState, self).__init__(
            name=name,
            position=position,
        )

    def deepcopy(self):
        return IdObjectState(self.id, self.name, self.position)

    def __eq__(self, other):
        return (
            isinstance(other, IdObjectState)
            and self.name == other.name
            and self.position == other.position
        )

    @classmethod
    def from_dict(cls, obj_dict):
        obj_dict = copy.deepcopy(obj_dict)
        return IdObjectState(**obj_dict)

    def is_valid(self):
        return self.name in [
            "dirty_plate",
            "meat",
            "dish",
            "chicken",
            "onion",
            "steak",
            "boiled_chicken",
            "steak_onion",
            "boiled_chicken_onion",
        ]


class ChickenState(IdObjectState):
    def __init__(
        self,
        id,
        name,
        position,
        ingredients=[],
        cooking_tick=-1,
        cook_time=-1,
        **kwargs
    ):
        """
        Represents a soup object. An object becomes a soup the instant it is placed in a pot. The
        soup's recipe is a list of ingredient names used to create it. A soup's recipe is undetermined
        until it has begun cooking.

        position (tupe): (x, y) coordinates in the grid
        ingrdients (list(ObjectState)): Objects that have been used to cook this soup. Determiens @property recipe
        cooking_tick (int): How long the soup has been cooking for. -1 means cooking hasn't started yet
        cook_time(int): How long soup needs to be cooked, used only mostly for getting soup from dict with supplied cook_time, if None self.recipe.time is used
        """
        super(ChickenState, self).__init__(id, name, position)
        self._ingredients = ingredients
        self._cooking_tick = cooking_tick
        self._recipe = None
        self._cook_time = (
            cook_time if cook_time > 0 else Steakhouse_Recipe._chicken_time
        )

    def __eq__(self, other):
        return (
            isinstance(other, ChickenState)
            and self.name == other.name
            and self.position == other.position
            and self._cooking_tick == other._cooking_tick
            and all(
                [
                    this_i == other_i
                    for this_i, other_i in zip(self._ingredients, other._ingredients)
                ]
            )
        )

    def __hash__(self):
        ingredient_hash = hash(tuple([hash(i) for i in self._ingredients]))
        supercls_hash = super(ChickenState, self).__hash__()
        return hash((supercls_hash, self._cooking_tick, ingredient_hash))

    def __repr__(self):
        supercls_str = super(ChickenState, self).__repr__()
        ingredients_str = self._ingredients.__repr__()
        return "{}\nIngredients:\t{}\nCooking Tick:\t{}".format(
            supercls_str, ingredients_str, self._cooking_tick
        )

    def __str__(self):
        res = "{"
        for ingredient in sorted(self.ingredients):
            res += Steakhouse_Recipe.STR_REP[ingredient]
        if self.is_cooking:
            res += str(self._cooking_tick)
        elif self.is_ready:
            res += str("✓")
        return res

    @IdObjectState.position.setter
    def position(self, new_pos):
        self._position = new_pos
        for ingredient in self._ingredients:
            ingredient.position = new_pos

    @property
    def ingredients(self):
        return [ingredient.name for ingredient in self._ingredients]

    @property
    def is_cooking(self):
        return not self.is_idle and not self.is_ready

    @property
    def recipe(self):
        if self.is_idle:
            raise ValueError("Recipe is not determined until soup begins cooking")
        if not self._recipe:
            self._recipe = Steakhouse_Recipe(self.ingredients)
        return self._recipe

    @property
    def value(self):
        return self.recipe.value

    @property
    def cook_time(self):
        # used mostly when cook time is supplied by state dict
        if self._cook_time is not None:
            return self._cook_time
        else:
            return self.recipe.time

    @property
    def cook_time_remaining(self):
        return max(0, self._cook_time - self._cooking_tick)

    @property
    def is_ready(self):
        if self.is_idle:
            return False
        return self._cooking_tick >= self._cook_time

    @property
    def is_idle(self):
        return self._cooking_tick < 0

    @property
    def is_full(self):
        return (
            not self.is_idle
            or len(self.ingredients) == Steakhouse_Recipe.MAX_NUM_INGREDIENTS
        )

    def is_valid(self):
        if not all(
            [ingredient.position == self.position for ingredient in self._ingredients]
        ):
            return False
        if len(self.ingredients) > Steakhouse_Recipe.MAX_NUM_INGREDIENTS:
            return False
        return True

    def auto_finish(self):
        if len(self.ingredients) == 0:
            raise ValueError("Cannot finish chicken with no ingredients")
        self._cooking_tick = 0
        self._cooking_tick = self._cook_time

    def add_ingredient(self, ingredient):
        if not ingredient.name in Steakhouse_Recipe.ALL_INGREDIENTS:
            raise ValueError("Invalid ingredient")
        if self.is_full:
            raise ValueError("Reached maximum number of ingredients in recipe")
        ingredient.position = self.position
        self._ingredients.append(ingredient)

    def add_ingredient_from_str(self, ingredient_str):
        ingredient_obj = IdObjectState(ingredient_str, self.position)
        self.add_ingredient(ingredient_obj)

    def pop_ingredient(self):
        if not self.is_idle:
            raise ValueError(
                "Cannot remove an ingredient from this Chicken at this time"
            )
        if len(self._ingredients) == 0:
            raise ValueError("No ingredient to remove")
        return self._ingredients.pop()

    def begin_cooking(self):
        if not self.is_idle:
            raise ValueError("Cannot begin cooking this chicken soup at this time")
        if len(self.ingredients) == 0:
            raise ValueError(
                "Must add at least one ingredient to chicken soup before you can begin cooking"
            )
        self._cooking_tick = 0

    def cook(self):
        if self.is_idle:
            raise ValueError("Must begin cooking before advancing cook tick")
        if self.is_ready:
            raise ValueError("Cannot cook a soup that is already done")
        self._cooking_tick += 1

    def deepcopy(self):
        return ChickenState(
            self.id,
            self.name,
            self.position,
            [ingredient.deepcopy() for ingredient in self._ingredients],
            self._cooking_tick,
            self._cook_time,
        )

    def to_dict(self):
        info_dict = super(ChickenState, self).to_dict()
        ingrdients_dict = [ingredient.to_dict() for ingredient in self._ingredients]
        info_dict["_ingredients"] = ingrdients_dict
        info_dict["cooking_tick"] = self._cooking_tick
        info_dict["is_cooking"] = self.is_cooking
        info_dict["is_ready"] = self.is_ready
        info_dict["is_idle"] = self.is_idle
        info_dict["cook_time"] = -1 if self.is_idle else self._cook_time

        # This is for backwards compatibility w/ overcooked-demo
        # Should be removed once overcooked-demo is updated to use 'cooking_tick' instead of '_cooking_tick'
        info_dict["_cooking_tick"] = self._cooking_tick
        return info_dict

    @classmethod
    def from_dict(cls, obj_dict):
        obj_dict = copy.deepcopy(obj_dict)
        if obj_dict["name"] != "soup":
            return super(ChickenState, cls).from_dict(obj_dict)

        if "state" in obj_dict:
            # Legacy soup representation
            ingredient, num_ingredient, time = obj_dict["state"]
            cooking_tick = -1 if time == 0 else time
            finished = time >= Steakhouse_Recipe._chicken_time
            if ingredient == Steakhouse_Recipe.CHICKEN:
                return ChickenState.get_soup(
                    obj_dict["position"],
                    num_chicken=num_ingredient,
                    cooking_tick=cooking_tick,
                    finished=finished,
                )
        ingredients_objs = [
            IdObjectState.from_dict(ing_dict) for ing_dict in obj_dict["_ingredients"]
        ]
        obj_dict["ingredients"] = ingredients_objs
        return cls(**obj_dict)

    @classmethod
    def get_chicken(
        cls, position, num_chicken=0, cooking_tick=-1, finished=False, **kwargs
    ):
        if num_chicken < 0:
            raise ValueError("Number of active ingredients must be positive")
        if num_chicken > Steakhouse_Recipe.MAX_NUM_INGREDIENTS:
            raise ValueError("Too many ingredients specified for this soup")
        if cooking_tick >= 0 and num_chicken == 0:
            raise ValueError("_cooking_tick must be -1 for empty soup")
        if finished and num_chicken == 0:
            raise ValueError("Empty soup cannot be finished")
        chicken = [
            IdObjectState(Steakhouse_Recipe.CHICKEN, position)
            for _ in range(num_chicken)
        ]
        ingredients = chicken
        soup = cls(position, ingredients, cooking_tick)
        if finished:
            soup.auto_finish()
        return soup


class PlateState(IdObjectState):
    def __init__(self, id, name, position, rinse_total=3, rinse_count=-1, **kwargs):
        super(PlateState, self).__init__(id, name, position)
        self._cook_time = rinse_total
        self._cooking_tick = rinse_count

    def __eq__(self, other):
        return (
            isinstance(other, PlateState)
            and self.id == other.id
            and self.name == other.name
            and self.position == other.position
            and self._cooking_tick == other._cooking_tick
        )

    def __hash__(self):
        supercls_hash = super(PlateState, self).__hash__()
        return hash((supercls_hash, self._cooking_tick))

    def __repr__(self):
        supercls_str = super(PlateState, self).__repr__()
        return "{}\nRinse Count:\t{}".format(supercls_str, self._cooking_tick)

    def __str__(self):
        res = "{"
        if self.is_rinsing:
            res += str(self._cooking_tick)
        elif self.is_ready:
            res += str("✓")
        return res

    @ObjectState.position.setter
    def position(self, new_pos):
        self._position = new_pos

    @property
    def is_rinsing(self):
        return not self.is_idle and not self.is_ready

    @property
    def cook_time(self):
        # used mostly when cook time is supplied by state dict
        if self._cook_time is not None:
            return self._cook_time
        else:
            return 2

    def is_valid(self):
        return self.name in ["clean_plate", "dirty_plate"]

    @property
    def rinse_time_remaining(self):
        return max(0, self._cook_time - self._cooking_tick)

    @property
    def is_ready(self):
        if self.is_idle:
            return False
        return self._cooking_tick >= self._cook_time

    @property
    def is_idle(self):
        return self._cooking_tick < 0

    @property
    def is_full(self):
        return not self.is_idle

    def auto_finish(self):
        self._cooking_tick = 0
        self._cooking_tick = self._cook_time

    @IdObjectState.position.setter
    def position(self, new_pos):
        self._position = new_pos

    def begin_rinsing(self):
        if not self.is_idle:
            raise ValueError("Cannot begin rinse at this time")
        self._cooking_tick = 0

    def rinse(self):
        if self.is_idle:
            raise ValueError("Must begin rinsing before advancing rinse tick")
        if self.is_ready:
            raise ValueError("Cannot rinse a plate that is already done")
        self._cooking_tick += 1

    def deepcopy(self):
        return PlateState(
            self.id, self.name, self.position, self._cook_time, self._cooking_tick
        )

    def to_dict(self):
        info_dict = super(PlateState, self).to_dict()
        info_dict["rinse_count"] = self._cooking_tick
        info_dict["is_ready"] = self.is_ready
        info_dict["is_idle"] = self.is_idle
        info_dict["rinse_total"] = self._cook_time
        return info_dict

    @classmethod
    def from_dict(cls, obj_dict):
        obj_dict = copy.deepcopy(obj_dict)
        if obj_dict["name"] != "clean_plate" or obj_dict["name"] != "dirty_plate":
            return super(SoupState, cls).from_dict(obj_dict)

    # @classmethod
    # def get_plate(cls, position, rinse_total=2):
    #     return cls(position, rinse_total)

    # @classmethod
    # def get_plate(cls, id, position, rinse_total=3, rinse_count=-1, finished=False, **kwargs):
    #     plate = cls(id, "plate", position, rinse_total=rinse_total, rinse_count=rinse_count)
    #     if finished:
    #         plate.auto_finish()
    #     return plate


class SteakState(SoupState):
    def __init__(
        self, id, name, position, ingredients=[], cooking_tick=-1, cook_time=-1
    ):
        super(SteakState, self).__init__(position, ingredients)
        self.id = id
        self.name = name
        self._cooking_tick = cooking_tick
        self._cook_time = cook_time if cook_time > 0 else Steakhouse_Recipe._steak_time

    def __eq__(self, other):
        return (
            isinstance(other, SteakState)
            and self.name == other.name
            and self.position == other.position
            and self._cooking_tick == other._cooking_tick
            and all(
                [
                    this_i == other_i
                    for this_i, other_i in zip(self._ingredients, other._ingredients)
                ]
            )
        )

    def __hash__(self):
        ingredient_hash = hash(tuple([hash(i) for i in self._ingredients]))
        supercls_hash = super(SteakState, self).__hash__()
        return hash((supercls_hash, self._cooking_tick, ingredient_hash))

    def __repr__(self):
        supercls_str = super(SteakState, self).__repr__()
        ingredients_str = self._ingredients.__repr__()
        return "{}\nIngredients:\t{}\nCooking Tick:\t{}".format(
            supercls_str, ingredients_str, self._cooking_tick
        )

    def __str__(self):
        res = "{"
        for ingredient in sorted(self.ingredients):
            res += Steakhouse_Recipe.STR_REP[ingredient]
        if self.is_cooking:
            res += str(self._cooking_tick)
        elif self.is_ready:
            res += str("✓")
        return res

    def is_valid(self):
        return self.name in ["steak"]

    @IdObjectState.position.setter
    def position(self, new_pos):
        self._position = new_pos
        for ingredient in self._ingredients:
            ingredient.position = new_pos

    @property
    def ingredients(self):
        return [ingredient.name for ingredient in self._ingredients]

    @property
    def is_cooking(self):
        return not self.is_idle and not self.is_ready

    @property
    def cook_time(self):
        return self._cook_time

    @property
    def is_ready(self):
        if self.is_idle:
            return False
        return self._cooking_tick >= self._cook_time

    @property
    def is_idle(self):
        return self._cooking_tick < 0

    @property
    def is_full(self):
        return (
            not self.is_idle
            or len(self.ingredients) == Steakhouse_Recipe.MAX_NUM_INGREDIENTS
        )

    def auto_finish(self):
        if len(self.ingredients) == 0:
            raise ValueError("Cannot finish steak with no ingredients")
        self._cooking_tick = 0
        self._cooking_tick = self._cook_time

    def add_ingredient(self, ingredient):
        if not ingredient.name in Steakhouse_Recipe.ALL_INGREDIENTS:
            raise ValueError("Invalid ingredient")
        if self.is_full:
            raise ValueError("Reached maximum number of ingredients in recipe")
        ingredient.position = self.position
        self._ingredients.append(ingredient)

    def add_ingredient_from_str(self, ingredient_str):
        ingredient_obj = IdObjectState(None, ingredient_str, self.position)
        self.add_ingredient(ingredient_obj)

    def pop_ingredient(self):
        if not self.is_idle:
            raise ValueError("Cannot remove an ingredient from this steak at this time")
        if len(self._ingredients) == 0:
            raise ValueError("No ingredient to remove")
        return self._ingredients.pop()

    def begin_cooking(self):
        if not self.is_idle:
            raise ValueError("Cannot begin cooking this steak at this time")
        if len(self.ingredients) == 0:
            raise ValueError(
                "Must add at least one ingredient to steak before you can begin cooking"
            )
        self._cooking_tick = 0

    def cook(self):
        if self.is_idle:
            raise ValueError("Must begin cooking before advancing cook tick")
        if self.is_ready:
            raise ValueError("Cannot cook a soup that is already done")
        self._cooking_tick += 1

    @classmethod
    def get_steak(cls, id, position, num_meat=1, cooking_tick=-1, finished=False, **kwargs):
        if num_meat < 0:
            raise ValueError("Number of active ingredients must be positive")
        if num_meat > Recipe.MAX_NUM_INGREDIENTS:
            raise ValueError("Too many ingredients specified for steak")
        if cooking_tick >= 0 and num_meat == 0:
            raise ValueError("_cooking_tick must be -1 for empty grill")
        if finished and num_meat == 0:
            raise ValueError("Empty grill cannot be finished")
        meats = [ObjectState(Steakhouse_Recipe.MEAT, position) for _ in range(num_meat)]
        ingredients = meats
        steak = cls(id, "steak", position, ingredients=ingredients, cooking_tick=cooking_tick)
        if finished:
            steak.auto_finish()
        return steak

    def to_dict(self):
        info_dict = super(SteakState, self).to_dict()
        ingrdients_dict = [ingredient.to_dict() for ingredient in self._ingredients]
        info_dict["_ingredients"] = ingrdients_dict
        info_dict["cooking_tick"] = self._cooking_tick
        info_dict["is_cooking"] = self.is_cooking
        info_dict["is_ready"] = self.is_ready
        info_dict["is_idle"] = self.is_idle
        info_dict["cook_time"] = -1 if self.is_idle else self._cook_time

        # This is for backwards compatibility w/ overcooked-demo
        # Should be removed once overcooked-demo is updated to use 'cooking_tick' instead of '_cooking_tick'
        info_dict["_cooking_tick"] = self._cooking_tick
        return info_dict

    @classmethod
    def from_dict(cls, obj_dict):
        obj_dict = copy.deepcopy(obj_dict)
        if obj_dict["name"] != "steak":
            return super(SteakState, cls).from_dict(obj_dict)

        if "state" in obj_dict:
            # Legacy soup representation
            ingredient, num_ingredient, time = obj_dict["state"]
            cooking_tick = -1 if time == 0 else time
            finished = time >= Steakhouse_Recipe._steak_time
            return SteakState.get_steak(
                obj_dict["id"],
                obj_dict["position"],
                num_meat=num_ingredient,
                cooking_tick=cooking_tick,
                finished=finished,
            )

        ingredients_objs = [
            IdObjectState.from_dict(ing_dict) for ing_dict in obj_dict["_ingredients"]
        ]
        obj_dict["ingredients"] = ingredients_objs
        return cls(**obj_dict)

    def deepcopy(self):
        return SteakState(
            self.id,
            self.name,
            self.position,
            [ingredient.deepcopy() for ingredient in self._ingredients],
            self._cooking_tick,
        )


class GarnishState(SoupState):
    def __init__(self, id, name, position, ingredients=[], chop_count=-1, chop_time=2):
        super(GarnishState, self).__init__(position, ingredients)
        self.id = id
        self.name = name
        self._cooking_tick = chop_count
        self._cook_time = chop_time

    def __eq__(self, other):
        return (
            isinstance(other, GarnishState)
            and self.name == other.name
            and self.position == other.position
            and self._cooking_tick == other._cooking_tick
            and all(
                [
                    this_i == other_i
                    for this_i, other_i in zip(self._ingredients, other._ingredients)
                ]
            )
        )

    def __hash__(self):
        ingredient_hash = hash(tuple([hash(i) for i in self._ingredients]))
        supercls_hash = super(GarnishState, self).__hash__()
        return hash((supercls_hash, self._cooking_tick, ingredient_hash))

    def __repr__(self):
        supercls_str = super(GarnishState, self).__repr__()
        ingredients_str = self._ingredients.__repr__()
        return "{}\nIngredients:\t{}\nCooking Tick:\t{}".format(
            supercls_str, ingredients_str, self._cooking_tick
        )

    def is_valid(self):
        return self.name in ["garnish"]

    def begin_chop(self):
        if not self.is_idle:
            raise ValueError("Cannot begin rinse at this time")
        self._cooking_tick = 0

    def chop(self):
        if self.is_ready:
            raise ValueError("Cannot cook a soup that is already done")
        self._cooking_tick += 1

    @IdObjectState.position.setter
    def position(self, new_pos):
        self._position = new_pos
        for ingredient in self._ingredients:
            ingredient.position = new_pos

    def add_ingredient_from_str(self, ingredient_str):
        ingredient_obj = IdObjectState(None, ingredient_str, self.position)
        self.add_ingredient(ingredient_obj)

    @classmethod
    def get_garnish(
        cls, id, position, num_onion=1, chop_count=-1, finished=False, **kwargs
    ):
        if num_onion < 0:
            raise ValueError("Number of active ingredients must be positive")
        if num_onion > Recipe.MAX_NUM_INGREDIENTS:
            raise ValueError("Too many ingredients specified for garnish")
        if chop_count >= 0 and num_onion == 0:
            raise ValueError("_chop_count must be -1 for empty board")
        if finished and num_onion == 0:
            raise ValueError("Empty board cannot be finished")
        onions = [
            ObjectState(Steakhouse_Recipe.ONION, position) for _ in range(num_onion)
        ]
        ingredients = onions
        garnish = cls(id, "garnish", position, ingredients=ingredients, chop_count=chop_count)
        if finished:
            garnish.auto_finish()
        return garnish

    def deepcopy(self):
        return GarnishState(
            self.id,
            self.name,
            self.position,
            [ingredient.deepcopy() for ingredient in self._ingredients],
            self._cooking_tick,
        )

    def to_dict(self):
        info_dict = super(GarnishState, self).to_dict()
        ingrdients_dict = [ingredient.to_dict() for ingredient in self._ingredients]
        info_dict["_ingredients"] = ingrdients_dict
        info_dict["cooking_tick"] = self._cooking_tick
        info_dict["is_cooking"] = self.is_cooking
        info_dict["is_ready"] = self.is_ready
        info_dict["is_idle"] = self.is_idle
        info_dict["cook_time"] = -1 if self.is_idle else self._cook_time

        # This is for backwards compatibility w/ overcooked-demo
        # Should be removed once overcooked-demo is updated to use 'cooking_tick' instead of '_cooking_tick'
        info_dict["_cooking_tick"] = self._cooking_tick
        return info_dict

    @classmethod
    def from_dict(cls, obj_dict):
        obj_dict = copy.deepcopy(obj_dict)
        if obj_dict["name"] != "garnish":
            return super(GarnishState, cls).from_dict(obj_dict)

        if "state" in obj_dict:
            # Legacy soup representation
            ingredient, num_ingredient, time = obj_dict["state"]
            cooking_tick = -1 if time == 0 else time
            finished = time >= 10
            return GarnishState.get_garnish(
                obj_dict["id"],
                obj_dict["position"],
                num_onion=num_ingredient,
                cooking_tick=cooking_tick,
                finished=finished,
            )

        ingredients_objs = [
            IdObjectState.from_dict(ing_dict) for ing_dict in obj_dict["_ingredients"]
        ]
        obj_dict["ingredients"] = ingredients_objs
        return cls(**obj_dict)


class SteakhouseState(OvercookedState):
    def __init__(
        self,
        players,
        objects,
        bonus_orders=[],
        all_orders=[],
        complete_orders=[],
        order_display_list=[],
        order_list=[],
        timestep=0,
        obj_count=None,
        **kwargs
    ):
        self.obj_count = obj_count if obj_count is not None else len(objects)
        all_orders = [Steakhouse_Recipe.from_dict(order) for order in all_orders]
        self._all_orders = all_orders
        for pos, obj in objects.items():
            assert obj.position == pos
        self.players = tuple(players)
        self.objects = objects
        self._bonus_orders = bonus_orders
        self._complete_orders = complete_orders
        self._order_display_list = order_display_list
        self.order_list = order_list
        self.timestep = timestep
        # assert len(set(self.bonus_orders)) == len(
        #     self.bonus_orders
        # ), "Bonus orders must not have duplicates"
        assert len(set(self.all_orders)) == len(
            self.all_orders
        ), "All orders must not have duplicates"
        # assert set(self.bonus_orders).issubset(
        #     set(self.all_orders)
        # ), "Bonus orders must be a subset of all orders"

    def deepcopy(self):
        return SteakhouseState(
            players=[player.deepcopy() for player in self.players],
            objects={pos: obj.deepcopy() for pos, obj in self.objects.items()},
            bonus_orders=[order for order in self._bonus_orders],
            all_orders=[order.to_dict() for order in self.all_orders],
            timestep=self.timestep,
            obj_count=self.obj_count,
            order_list=[order for order in self.order_list],
            order_display_list=[order for order in self._order_display_list],
        )

    def time_independent_equal(self, other):
        order_lists_equal = self.all_orders == other.all_orders

        return (
            isinstance(other, SteakhouseState)
            and self.players == other.players
            and set(self.objects.items()) == set(other.objects.items())
            and order_lists_equal
        )

    def to_dict(self):
        return {
            "players": [p.to_dict() for p in self.players],
            "objects": [obj.to_dict() for obj in self.objects.values()],
            "bonus_orders": [order for order in self.bonus_orders],
            "all_orders": [order.to_dict() for order in self.all_orders],
            "timestep": self.timestep,
        }

    @property
    def all_orders(self):
        return (
            sorted(self._all_orders)
            if self._all_orders
            else sorted(Steakhouse_Recipe.ALL_RECIPES)
        )

    @property
    def curr_order(self):
        return self.order_list[0]

    @property
    def num_orders_remaining(self):
        return len(self.order_list)


    @classmethod
    def from_players_pos_and_or(
        cls,
        players_pos_and_or,
        bonus_orders=[],
        all_orders=[],
        order_list=[],
        order_display_list=[],
    ):
        """
        Make a dummy OvercookedState with no objects based on the passed in player
        positions and orientations and order list
        """
        return cls(
            [
                PlayerState(*player_pos_and_or)
                for player_pos_and_or in players_pos_and_or
            ],
            objects={},
            bonus_orders=bonus_orders,
            all_orders=all_orders,
            order_list=order_list,
            order_display_list=order_list,
        )
    
    @classmethod
    def from_player_positions(
        cls,
        player_positions,
        bonus_orders=[],
        all_orders=[],
        order_list=[],
        order_display_list=[],
    ):
        """
        Make a dummy OvercookedState with no objects and with players facing
        North based on the passed in player positions and order list
        """
        dummy_pos_and_or = [(pos, Direction.NORTH) for pos in player_positions]
        return cls.from_players_pos_and_or(
            dummy_pos_and_or, bonus_orders, all_orders, order_list, order_display_list
        )

    @staticmethod
    def from_dict(state_dict, obj_count=0):
        state_dict = copy.deepcopy(state_dict)
        state_dict["players"] = [
            PlayerState.from_dict(p) for p in state_dict["players"]
        ]
        object_list = [IdObjectState.from_dict(o) for o in state_dict["objects"]]
        state_dict["objects"] = {ob.position: ob for ob in object_list}
        return SteakhouseState(**state_dict, obj_count=obj_count)

    # below methods ported from ICAROS qd-humans framework for RL agents

    # def print_player_workload(
    #     self,
    # ):
    #     for idx, player in enumerate(self.players):
    #         logger.info(f"Player {idx + 1}")
    #         player.print_workload()

    def get_player_workload(
        self,
    ):
        workloads = []
        for idx, player in enumerate(self.players):
            workloads.append(player.get_workload())
        return workloads

    def cal_concurrent_active_frequency(
        self,
    ):
        """Proportion of time in which both agents are active (\in [0,1])"""
        concurrent_active_log = self.cal_concurrent_active_log()
        return np.mean(concurrent_active_log)

    def cal_concurrent_active_sum(
        self,
    ):
        concurrent_active_log = self.cal_concurrent_active_log()
        res = np.sum(concurrent_active_log)

        return res

    def cal_concurrent_active_log(
        self,
    ):
        active_logs = self.get_player_active_log()
        if len(active_logs[0]) == 0:
            return []

        return np.array(active_logs[0]) & np.array(active_logs[1])

    def get_player_active_log(
        self,
    ):
        active_log = []
        for idx, player in enumerate(self.players):
            active_log.append(player.active_log)
        return active_log

    def cal_mean_stuck_time(
        self,
    ):
        """Proportion of time in which both agents are stuck (\in [0,1])"""
        stuck_logs = self.get_player_stuck_log()
        return np.mean(stuck_logs[0])

    def cal_total_stuck_time(
        self,
    ):
        stuck_logs = self.get_player_stuck_log()
        res = sum(stuck_logs[0])
        return res

    def get_player_stuck_log(
        self,
    ):
        stuck_log = []
        for idx, player in enumerate(self.players):
            stuck_log.append(player.stuck_log)
        return stuck_log


def dishname2ingradient(dish_name):
    # map dish_name to its ingredient, for example, steak_onion_dish to {"ingredients" : ["meat","onion"]},
    if dish_name == "steak_dish":
        return {"ingredients": ["meat"]}
    elif dish_name == "boiled_chicken_dish":
        return {"ingredients": ["chicken"]}
    elif dish_name == "steak_onion_dish":
        return {"ingredients": ["meat", "onion"]}
    elif dish_name == "boiled_chicken_onion_dish":
        return {"ingredients": ["chicken", "onion"]}


def ingradient2dishname(ingradient):
    # map ingradient to its dish_name, for example, {"ingredients" : ["meat","onion"]} to steak_onion_dish
    if ingradient == ["meat"]:
        return "steak_dish"
    elif ingradient == ["chicken"]:
        return "boiled_chicken_dish"
    elif ingradient == ["meat", "onion"]:
        return "steak_onion_dish"
    elif ingradient == ["chicken", "onion"]:
        return "boiled_chicken_onion_dish"


DISH_TYPES = [
    "steak_dish",
    "boiled_chicken_dish",
    "steak_onion_dish",
    "boiled_chicken_onion_dish",
]

EVENT_TYPES = [
    # Onion events
    "onion_pickup",
    "useful_onion_pickup",
    "onion_drop",
    "useful_onion_drop",
    "potting_onion",
    # Meat events
    "meat_pickup",
    "useful_meat_pickup",
    "meat_drop",
    "useful_meat_drop",
    # chicken events,
    "chicken_pickup",
    "useful_chicken_pickup",
    "chicken_drop",
    "useful_chicken_drop",
    "potting_chicken",
    # Dish events
    "useful_steak_pickup",
    "useful_steak_drop",
    "steak_cooking",
    "dish_pickup",
    "steak_pickup",
    "boiled_chicken_pickup",
    "boiled_chicken_drop",
    "useful_boiled_chicken_pickup",
    "useful_dish_pickup",
    "dish_drop",
    "steak_drop",
    "boiled_chicken_onion_drop",
    "useful_dish_drop",
    "useful_steak_drop",
    "useful_boiled_chicken_drop",
    "dish_delivery",
    "steak_onion_pickup",
    "boiled_chicken_onion_pickup",
    "useful_steak_onion_pickup",
    "useful_boiled_chicken_onion_pickup",
    "steak_onion_drop",
    "boiled_onion_drop",
    "useful_steak_onion_drop",
    "useful_boiled_chicken_onion_drop",
    "steak_onion_dish_delivery",
    "boiled_chicken_onion_delivery",
    "steak_dish_delivery",
    "boiled_chicken_delivery",
    # Soup events
    "soup_pickup",
    "soup_delivery",
    "soup_drop",
    # Potting events
    "optimal_onion_potting",
    "optimal_tomato_potting",
    "viable_onion_potting",
    "viable_tomato_potting",
    "catastrophic_onion_potting",
    "catastrophic_tomato_potting",
    "useless_onion_potting",
    "useless_tomato_potting",
    # Chopping events
    "chop_onion",
    "onion_chopping",
    # Rinsing events
    "plate_rinsing",
    "dirty_plate_drop",
    "dirty_plate_pickup",
    "rinse_dirty_plate",
    "clean_plate_pickup",
    "useful_clean_plate_pickup",
]


class SteakhouseGridworld(OvercookedGridworld):
    def __init__(
        self,
        terrain,
        start_player_positions,
        start_all_orders=None,
        order_list=None,
        order_display_list=None,
        bonus_list=None,
        cook_time=10,
        num_items_for_steak=1,
        num_items_for_chicken=1,
        num_items_for_soup=3,
        chop_time=3,
        in_order_delivery_reward=10,
        non_order_delivery_reward=-10,
        delivery_reward=5,
        rew_shaping_params=None,
        layout_name="unnamed_layout",
        object_id_dict={},
        enable_same_cell=False,
        **kwargs
    ):
        super().__init__(
            terrain=terrain,
            start_player_positions=start_player_positions,
            start_all_orders=start_all_orders,
            cook_time=cook_time,
            num_items_for_soup=num_items_for_soup,
            delivery_reward=delivery_reward,
            rew_shaping_params=rew_shaping_params,
            layout_name=layout_name,
        )
        self.steak_cook_time = cook_time
        self.chop_time = chop_time
        self.object_id_dict = object_id_dict
        self.num_items_for_steak = num_items_for_steak
        self.num_items_for_chicken = num_items_for_chicken
        self.order_list = order_list
        self.order_display_list = order_display_list
        self.delivery_reward = delivery_reward
        self.in_order_delivery_reward = in_order_delivery_reward
        self.non_order_delivery_reward = non_order_delivery_reward
        self.enable_same_cell = enable_same_cell

        self._configure_steakhouse_recipes(
            start_all_orders, num_items_for_chicken, num_items_for_steak, **kwargs
        )
        self.start_all_orders = (
            [r.to_dict() for r in Steakhouse_Recipe.ALL_RECIPES]
            if not start_all_orders
            else start_all_orders
        )

    @staticmethod
    def from_layout_name(layout_name, **params_to_overwrite):
        """
        Generates a OvercookedGridworld instance from a layout file.

        One can overwrite the default mdp configuration using partial_mdp_config.
        """
        params_to_overwrite = params_to_overwrite.copy()
        base_layout_params = read_layout_dict(layout_name)

        grid = base_layout_params["grid"]
        del base_layout_params["grid"]
        base_layout_params["layout_name"] = layout_name
        if "start_state" in base_layout_params:
            base_layout_params["start_state"] = SteakhouseState.from_dict(
                base_layout_params["start_state"]
            )

        # Clean grid
        grid = [layout_row.strip() for layout_row in grid.split("\n")]
        return SteakhouseGridworld.from_grid(
            grid, base_layout_params, params_to_overwrite
        )

    @staticmethod
    def _assert_valid_grid(grid):
        """Raises an AssertionError if the grid is invalid.

        grid:  A sequence of sequences of spaces, representing a grid of a
        certain height and width. grid[y][x] is the space at row y and column
        x. A space must be either 'X' (representing a counter), ' ' (an empty
        space), 'O' (onion supply), 'P' (pot), 'D' (dish supply), 'S' (serving
        location), '1' (player 1) and '2' (player 2).
        """
        height = len(grid)
        width = len(grid[0])

        # Make sure the grid is not ragged
        assert all(len(row) == width for row in grid), "Ragged grid"

        # Borders must not be free spaces
        def is_not_free(c):
            return c in "XOPDCWBSGTM"

        for y in range(height):
            assert is_not_free(grid[y][0]), "Left border must not be free"
            assert is_not_free(grid[y][-1]), "Right border must not be free"
        for x in range(width):
            assert is_not_free(grid[0][x]), "Top border must not be free"
            assert is_not_free(grid[-1][x]), "Bottom border must not be free"

        all_elements = [element for row in grid for element in row]
        digits = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
        layout_digits = [e for e in all_elements if e in digits]
        num_players = len(layout_digits)
        assert num_players > 0, "No players (digits) in grid"
        layout_digits = list(sorted(map(int, layout_digits)))
        assert layout_digits == list(
            range(1, num_players + 1)
        ), "Some players were missing"
        # TODO: change this to allow more terrain, inherite.
        assert all(
            c in "XOPDSTWBMCG123456789 " for c in all_elements
        ), "Invalid character in grid"
        assert all_elements.count("1") == 1, "'1' must be present exactly once"
        assert all_elements.count("D") >= 1, "'D' must be present at least once"
        assert all_elements.count("S") >= 1, "'S' must be present at least once"
        # assert all_elements.count("P") >= 1, "'P' must be present at least once"
        # assert (
        #     all_elements.count("G") >= 1
        # ), "'G' must be present at least once"
        # assert (
        #     all_elements.count("M") >= 1
        # ), "'M' must be present at least once"

    @staticmethod
    def from_grid(
        layout_grid, base_layout_params={}, params_to_overwrite={}, debug=False
    ):
        """
        Returns instance of OvercookedGridworld with terrain and starting
        positions derived from layout_grid.
        One can override default configuration parameters of the mdp in
        partial_mdp_config.
        """
        mdp_config = copy.deepcopy(base_layout_params)

        layout_grid = [[c for c in row] for row in layout_grid]
        SteakhouseGridworld._assert_valid_grid(layout_grid)

        if "layout_name" not in mdp_config:
            layout_name = "|".join(["".join(line) for line in layout_grid])
            mdp_config["layout_name"] = layout_name

        player_positions = [None] * 9
        for y, row in enumerate(layout_grid):
            for x, c in enumerate(row):
                if c in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                    layout_grid[y][x] = " "

                    # -1 is to account for fact that player indexing starts from 1 rather than 0
                    assert (
                        player_positions[int(c) - 1] is None
                    ), "Duplicate player in grid"
                    player_positions[int(c) - 1] = (x, y)

        num_players = len([x for x in player_positions if x is not None])
        player_positions = player_positions[:num_players]

        # After removing player positions from grid we have a terrain mtx
        mdp_config["terrain"] = layout_grid
        mdp_config["start_player_positions"] = player_positions

        for k, v in params_to_overwrite.items():
            curr_val = mdp_config.get(k, None)
            if debug:
                print(
                    "Overwriting mdp layout standard config value {}:{} -> {}".format(
                        k, curr_val, v
                    )
                )
            mdp_config[k] = v

        return SteakhouseGridworld(**mdp_config)

    def _configure_steakhouse_recipes(
        self, start_all_orders, num_items_for_chicken, num_items_for_steak, **kwargs
    ):
        self.recipe_config = {
            "num_items_for_chicken": num_items_for_chicken,
            "num_items_for_steak": num_items_for_steak,
            "all_orders": start_all_orders,
            **kwargs,
        }
        Steakhouse_Recipe.configure(self.recipe_config)

    #####################
    # BASIC CLASS UTILS #
    #####################

    def __eq__(self, other):
        return (
            np.array_equal(self.terrain_mtx, other.terrain_mtx)
            and self.start_player_positions == other.start_player_positions
            and self.start_all_orders == other.start_all_orders
            and self.steak_cook_time == other.steak_cook_time
            and self.delivery_reward == other.delivery_reward
            and self.in_order_delivery_reward == other.in_order_delivery_reward
            and self.non_order_delivery_reward == other.non_order_delivery_reward
            and self.reward_shaping_params == other.reward_shaping_params
            and self.layout_name == other.layout_name
            and self.enable_same_cell == other.enable_same_cell
        )

    def copy(self):
        return SteakhouseGridworld(
            terrain=self.terrain_mtx.copy(),
            start_player_positions=self.start_player_positions,
            start_all_orders=None
            if self.start_all_orders is None
            else list(self.start_all_orders),
            cook_time=self.steak_cook_time,
            delivery_reward=self.delivery_reward,
            in_order_delivery_reward=self.in_order_delivery_reward,
            non_order_delivery_reward=self.non_order_delivery_reward,
            rew_shaping_params=copy.deepcopy(self.reward_shaping_params),
            layout_name=self.layout_name,
            object_id_dict=copy.deepcopy(self.object_id_dict),
            enable_same_cell = self.enable_same_cell
        )

    @property
    def mdp_params(self):
        return {
            "layout_name": self.layout_name,
            "terrain": self.terrain_mtx,
            "start_player_positions": self.start_player_positions,
            "start_all_orders": self.start_all_orders,
            "cook_time": self.soup_cook_time,
            "delivery_reward": self.delivery_reward,
            "in_order_delivery_reward": self.in_order_delivery_reward,
            "non_order_delivery_reward": self.non_order_delivery_reward,
            "rew_shaping_params": copy.deepcopy(self.reward_shaping_params),
            "enable_same_cell": self.enable_same_cell,
        }

    ##############
    # GAME LOGIC #
    ##############

    def get_actions(self, state):
        """
        Returns the list of lists of valid actions for 'state'.

        The ith element of the list is the list of valid actions that player i
        can take.
        """
        self._check_valid_state(state)
        return [
            self._get_player_actions(state, i)
            for i in range(len(state.players))
        ]
    
    def get_standard_start_state(self):
        if self.start_state:
            return self.start_state
        start_state = SteakhouseState.from_player_positions(
            self.start_player_positions,
            all_orders=self.start_all_orders,
            order_list=self.order_list,
        )
        return start_state

    def rand_pos_start_state_fn(self):
        valid_positions = self.get_valid_joint_player_positions()
        start_pos = valid_positions[
            np.random.choice(len(valid_positions))
        ]
        start_state = SteakhouseState.from_player_positions(
            start_pos,
            bonus_orders=self.start_bonus_orders,
            all_orders=self.start_all_orders,
            order_list=self.order_list,
            order_display_list=self.order_display_list,
        )
        return start_state

    def get_random_objects_start_state_fn(self, random_start_pos=False, rnd_obj_prob_thresh=0.5):
        """
        Creates a function that returns a random start state with some key objects occupied.
        
        Args:
            random_start_pos (bool): Whether to randomize player starting positions
            rnd_obj_prob_thresh (float): Probability threshold for adding objects to the state
            
        Returns:
            A function that when called returns a SteakhouseState with random objects
        """
        def start_state_fn():
            self.object_id_dict = {}
            obj_count = len(self.object_id_dict)
            # Get random or fixed player positions
            if random_start_pos:
                valid_positions = self.get_valid_joint_player_positions()
                start_pos = valid_positions[
                    np.random.choice(len(valid_positions))
                ]
            else:
                start_pos = self.start_player_positions

            # Create base state
            start_state = SteakhouseState.from_player_positions(
                start_pos,
                bonus_orders=self.start_bonus_orders,
                all_orders=self.start_all_orders,
                order_list=self.order_list,
                order_display_list=self.order_display_list,
            )

            occupied_obj = random.randint(0, 1)
            # Add random objects to grills
            grill_locations = self.get_grill_locations()
            for grill_loc in grill_locations:
                p = np.random.rand()
                if p < rnd_obj_prob_thresh or occupied_obj == 0:
                    # Randomly decide if steak is cooking or ready
                    cooking_tick = 0 if np.random.rand() < 0.5 else 30
                    
                    # Create steak with random number of meat items
                    new_obj = SteakState.get_steak(
                        obj_count,
                        grill_loc,
                        cooking_tick=cooking_tick,
                    )
                    # start_state.objects[grill_loc] = new_obj
                    self.object_id_dict[obj_count] = new_obj
                    obj_count += 1
                    start_state.add_object(new_obj, grill_loc)

            # Add random objects to chopping boards
            chopping_board_locations = self.get_chopping_board_locations()
            for board_loc in chopping_board_locations:
                p = np.random.rand()
                if p < rnd_obj_prob_thresh:
                    # Randomly decide if garnish is being chopped or ready
                    chop_count = 0 if np.random.rand() < 0.5 else 2
                    finished = np.random.rand() < 0.3  # 30% chance of being ready
                    
                    # Create garnish with random number of onions
                    new_obj = GarnishState.get_garnish(
                        obj_count,
                        board_loc,
                        chop_count=chop_count if not finished else 2,
                    )
                    # start_state.objects[board_loc] = new_obj
                    self.object_id_dict[obj_count] = new_obj
                    obj_count += 1
                    start_state.add_object(new_obj, board_loc)

            # Add random objects to sinks (plates)
            sink_locations = self.get_sink_locations()
            for sink_loc in sink_locations:
                p = np.random.rand()
                if p < rnd_obj_prob_thresh or occupied_obj == 1:
                    # Randomly decide if plate is rinsing or ready
                    rinse_count = 0 if np.random.rand() < 0.5 else 2
                    finished = np.random.rand() < 0.3  # 30% chance of being ready
                    # Create plate with random rinse status
                    new_obj = PlateState(
                        obj_count,
                        "clean_plate",
                        sink_loc,
                        rinse_count=rinse_count if not finished else 2,
                    )
                    # start_state.objects[sink_loc] = new_obj
                    self.object_id_dict[obj_count] = new_obj
                    obj_count += 1
                    start_state.add_object(new_obj, sink_loc)

            start_state.obj_count = obj_count
            
            return start_state

        return start_state_fn
    
    def get_fixed_objects_start_state_fn1(self):
        """
        Creates a function that returns a random start state with some key objects occupied.
        
        Args:
            random_start_pos (bool): Whether to randomize player starting positions
            rnd_obj_prob_thresh (float): Probability threshold for adding objects to the state
            
        Returns:
            A function that when called returns a SteakhouseState with random objects
        """
        def start_state_fn():
            self.object_id_dict = {}
            obj_count = len(self.object_id_dict)
            # Get random or fixed player positions
            start_pos = ((2,6), (1,6))

            # Create base state
            start_state = SteakhouseState.from_player_positions(
                start_pos,
                bonus_orders=self.start_bonus_orders,
                all_orders=self.start_all_orders,
                order_list=self.order_list,
                order_display_list=self.order_display_list,
            )

            # Add random objects to grills
            grill_locations = self.get_grill_locations()
            for grill_loc in grill_locations:
                cooking_tick = 0 if np.random.rand() < 0.5 else 30
                
                # Create steak with random number of meat items
                new_obj = SteakState.get_steak(
                    obj_count,
                    grill_loc,
                    cooking_tick=cooking_tick,
                )
                # start_state.objects[grill_loc] = new_obj
                self.object_id_dict[obj_count] = new_obj
                obj_count += 1
                start_state.add_object(new_obj, grill_loc)

            # Add random objects to sinks (plates)
            sink_locations = self.get_sink_locations()
            for sink_loc in sink_locations:
                # Randomly decide if plate is rinsing or ready
                rinse_count = 0 if np.random.rand() < 0.5 else 2
                finished = np.random.rand() < 0.3  # 30% chance of being ready
                
                # Create plate with random rinse status
                new_obj = PlateState(
                    obj_count,
                    "clean_plate",
                    sink_loc,
                    rinse_count=rinse_count if not finished else 2,
                )
                # start_state.objects[sink_loc] = new_obj
                self.object_id_dict[obj_count] = new_obj
                obj_count += 1
                start_state.add_object(new_obj, sink_loc)

            start_state.obj_count = obj_count
            return start_state

        return start_state_fn
    
    def get_fixed_objects_start_state_fn2(self):
        """
        Creates a function that returns a random start state with some key objects occupied.
        
        Args:
            random_start_pos (bool): Whether to randomize player starting positions
            rnd_obj_prob_thresh (float): Probability threshold for adding objects to the state
            
        Returns:
            A function that when called returns a SteakhouseState with random objects
        """
        def start_state_fn():
            self.object_id_dict = {}
            obj_count = len(self.object_id_dict)
            # Get random or fixed player positions
            start_pos = ((5,1), (1,6))

            # Create base state
            start_state = SteakhouseState.from_player_positions(
                start_pos,
                bonus_orders=self.start_bonus_orders,
                all_orders=self.start_all_orders,
                order_list=self.order_list,
                order_display_list=self.order_display_list,
            )

            # Add random objects to grills
            grill_locations = self.get_grill_locations()
            for grill_loc in grill_locations:
                cooking_tick = 0 if np.random.rand() < 0.5 else 30
                
                # Create steak with random number of meat items
                new_obj = SteakState.get_steak(
                    obj_count,
                    grill_loc,
                    cooking_tick=cooking_tick,
                )
                # start_state.objects[grill_loc] = new_obj
                self.object_id_dict[obj_count] = new_obj
                obj_count += 1
                start_state.add_object(new_obj, grill_loc)

            # Add random objects to chopping boards
            chopping_board_locations = self.get_chopping_board_locations()
            for board_loc in chopping_board_locations:
                # Randomly decide if garnish is being chopped or ready
                chop_count = 0 if np.random.rand() < 0.5 else 2
                finished = np.random.rand() < 0.3  # 30% chance of being ready
                
                # Create garnish with random number of onions
                new_obj = GarnishState.get_garnish(
                    obj_count,
                    board_loc,
                    chop_count=chop_count if not finished else 2,
                )
                # start_state.objects[board_loc] = new_obj
                self.object_id_dict[obj_count] = new_obj
                obj_count += 1
                start_state.add_object(new_obj, board_loc)

            # # Add random objects to sinks (plates)
            # sink_locations = self.get_sink_locations()
            # for sink_loc in sink_locations:
            #     # Randomly decide if plate is rinsing or ready
            #     rinse_count = 0 if np.random.rand() < 0.5 else 2
            #     finished = np.random.rand() < 0.3  # 30% chance of being ready
                
            #     # Create plate with random rinse status
            #     new_obj = PlateState(
            #         obj_count,
            #         "clean_plate",
            #         sink_loc,
            #         rinse_count=rinse_count if not finished else 2,
            #     )
            #     # start_state.objects[sink_loc] = new_obj
            #     self.object_id_dict[obj_count] = new_obj
            #     obj_count += 1
            #     start_state.add_object(new_obj, sink_loc)

            start_state.obj_count = obj_count
            return start_state

        return start_state_fn

    def get_state_transition(
        self, state, joint_action, display_phi=False, motion_planner=None
    ):
        """Gets information about possible transitions for the action.

        Returns the next state, sparse reward and reward shaping.
        Assumes all actions are deterministic.

        NOTE: Sparse reward is given only when soups are delivered,
        shaped reward is given only for completion of subgoals
        (not soup deliveries).
        """
        events_infos = {event: [False] * self.num_players for event in EVENT_TYPES}
        assert not self.is_terminal(
            state
        ), "Trying to find successor of a terminal state: {}".format(state)

        for action, action_set in zip(joint_action, self.get_actions(state)):
            if action not in action_set:
                raise ValueError("Illegal action %s in state %s" % (action, state))

        new_state = state.deepcopy()
        # Resolve interacts first
        (
            sparse_reward_by_agent,
            shaped_reward_by_agent,
        ) = self.resolve_interacts(new_state, joint_action, events_infos)
        assert new_state.player_positions == state.player_positions
        assert new_state.player_orientations == state.player_orientations

        # Resolve player movements
        self.resolve_movement(new_state, joint_action)

        # Finally, environment effects
        self.step_environment_effects(new_state)

        # Additional dense reward logic
        # shaped_reward += self.calculate_distance_based_shaped_reward(state, new_state)
        infos = {
            "event_infos": events_infos,
            "sparse_reward_by_agent": sparse_reward_by_agent,
            "shaped_reward_by_agent": shaped_reward_by_agent,
        }
        if display_phi:
            assert (
                motion_planner is not None
            ), "motion planner must be defined if display_phi is true"
            infos["phi_s"] = self.potential_function(state, motion_planner)
            infos["phi_s_prime"] = self.potential_function(new_state, motion_planner)
        return new_state, infos
    
    def resolve_movement(self, state, joint_action):
        """Resolve player movement and deal with possible collisions"""
        (
            new_positions,
            new_orientations,
        ) = self.compute_new_positions_and_orientations(
            state.players, joint_action
        )
        for player_state, new_pos, new_o in zip(
            state.players, new_positions, new_orientations
        ):
            player_state.update_pos_and_or(new_pos, new_o)

    def resolve_interacts(self, new_state, joint_action, events_infos, rollout=True):
        """
        Resolve any INTERACT actions, if present.

        Currently if two players both interact with a terrain, we resolve player 1's interact
        first and then player 2's, without doing anything like collision checking.
        """
        pot_states = self.get_pot_states(new_state)
        # We divide reward by agent to keep track of who contributed
        sparse_reward, shaped_reward = [0] * self.num_players, [0] * self.num_players

        for player_idx, (player, action) in enumerate(
            zip(new_state.players, joint_action)
        ):
            if action != Action.INTERACT:
                continue

            pos, o = player.position, player.orientation
            i_pos = Action.move_in_direction(pos, o)
            terrain_type = self.get_terrain_type_at_pos(i_pos)
            if not rollout:
                obj_count = len(self.object_id_dict)
            else:
                obj_count = new_state.obj_count
            # NOTE: we always log pickup/drop before performing it, as that's
            # what the logic of determining whether the pickup/drop is useful assumes
            if terrain_type == "X":
                if player.has_object() and not new_state.has_object(i_pos):
                    obj_name = player.get_object().name
                    self.log_object_drop(
                        events_infos, new_state, obj_name, pot_states, player_idx
                    )

                    # Drop object on counter
                    obj = player.remove_object()
                    new_state.add_object(obj, i_pos)

                elif not player.has_object() and new_state.has_object(i_pos):
                    obj_name = new_state.get_object(i_pos).name
                    self.log_object_pickup(
                        events_infos, new_state, obj_name, pot_states, player_idx
                    )

                    # Pick up object from counter
                    obj = new_state.remove_object(i_pos)
                    player.set_object(obj)

                elif player.has_object() and new_state.has_object(i_pos):
                    obj_name = player.get_object().name
                    player_obj = player.remove_object()

                    # Pick up object from counter
                    self.log_object_pickup(
                        events_infos, new_state, obj_name, pot_states, player_idx
                    )
                    obj = new_state.remove_object(i_pos)
                    player.set_object(obj)

                    # Drop object on counter
                    self.log_object_drop(
                        events_infos, new_state, obj_name, pot_states, player_idx
                    )
                    new_state.add_object(player_obj, i_pos)

            elif terrain_type == "O" and player.held_object is None:
                # Onion pickup from dispenser
                self.log_object_pickup(
                    events_infos, new_state, "onion", pot_states, player_idx
                )
                new_o_id = obj_count
                o = IdObjectState(new_o_id, "onion", pos)
                if not rollout:
                    self.object_id_dict[new_o_id] = o
                obj_count += 1
                player.set_object(o)

                shaped_reward[player_idx] += self.reward_shaping_params[
                        "ONION_PICKUP_REWARD"]

                # player.num_ingre_held += 1

            elif terrain_type == "M" and player.held_object is None:
                # meat pickup from dispenser
                self.log_object_pickup(
                    events_infos, new_state, "meat", pot_states, player_idx
                )
                new_o_id = obj_count
                o = IdObjectState(new_o_id, "meat", pos)
                if not rollout:
                    self.object_id_dict[new_o_id] = o
                obj_count += 1
                player.set_object(o)

                shaped_reward[player_idx] += self.reward_shaping_params[
                        "MEAT_PICKUP_REWARD"]
                
                # player.num_ingre_held += 1

            elif (
                terrain_type == "C" and player.held_object is None
            ):  # chicken pickup from dispenser
                self.log_object_pickup(
                    events_infos, new_state, "chicken", pot_states, player_idx
                )

                new_o_id = obj_count
                o = IdObjectState(new_o_id, "chicken", pos)
                if not rollout:
                    self.object_id_dict[new_o_id] = o
                obj_count += 1
                player.set_object(o)
                shaped_reward[player_idx] += self.reward_shaping_params[
                        "CHICKEN_PICKUP_REWARD"]
                # player.num_ingre_held += 1


            elif terrain_type == "D" and player.held_object is None:
                self.log_object_pickup(
                    events_infos, new_state, "dirty_plate", pot_states, player_idx
                )
                # player.num_dirty_plate_held += 1

                # Give shaped reward if pickup is useful
                # if self.is_dirty_plate_pickup_useful(new_state, pot_states):
                shaped_reward[player_idx] += self.reward_shaping_params[
                        "DIRTY_PLATE_PICKUP_REWARD"]

                # Perform dirty plate pickup from dispenser
                new_o_id = obj_count
                o = IdObjectState(new_o_id, "dirty_plate", pos)
                if not rollout:
                    self.object_id_dict[new_o_id] = o
                obj_count += 1
                player.set_object(o)

            elif terrain_type == "W":
                if player.held_object is None:
                    # pick up clean plates
                    if self.plate_clean_at_location(new_state, i_pos):
                        self.log_object_pickup(
                            events_infos,
                            new_state,
                            "clean_plate",
                            pot_states,
                            player_idx,
                        )
                        obj = new_state.remove_object(i_pos)
                        player.set_object(obj)
                        # Give shaped reward if pickup is useful
                        # if self.is_dirty_plate_pickup_useful(new_state, pot_states):
                        shaped_reward[player_idx] += self.reward_shaping_params["CLEAN_PLATE_PICKUP_REWARD"]

                        # player.num_dirty_plate_held += 1

                    # rinse dirty plates
                    else:
                        if new_state.has_object(i_pos):
                            obj = new_state.get_object(i_pos)
                            if not obj.is_ready:
                                # print("rinse", obj, new_state)
                                obj.rinse()

                                events_infos["plate_rinsing"][player_idx] = True
                                
                                shaped_reward[player_idx] += self.reward_shaping_params["RINSE_DIRTY_PLATE"]

                else:  # sink is empty and put dirty plate
                    if (
                        player.get_object().name == "dirty_plate"
                        and not new_state.has_object(i_pos)
                    ):
                        obj_name = player.get_object().name
                        self.log_object_drop(
                            events_infos, new_state, obj_name, pot_states, player_idx
                        )

                        # Drop object on counter
                        obj = player.remove_object()
                        new_o_id = obj_count
                        new_obj = PlateState(new_o_id, "clean_plate", i_pos)
                        if not rollout:
                            self.object_id_dict[new_o_id] = new_obj
                        obj_count += 1
                        new_obj.begin_rinsing()
                        # print("begin rinsing", new_obj,new_state)
                        new_state.add_object(new_obj, i_pos)  # rinse time = 0

            elif terrain_type == "P" and player.has_object():
                # ready to pickup chicken from pot
                if (
                    player.get_object().name == "clean_plate"
                    and self.chicken_ready_at_location(new_state, i_pos)
                ):
                    self.log_object_pickup(
                        events_infos,
                        new_state,
                        "boiled_chicken",
                        pot_states,
                        player_idx,
                    )
                    # pickup chicken
                    player.remove_object()  # Remove the clean plate
                    obj = new_state.remove_object(i_pos)  # Get boiled chicken
                    player.set_object(obj)
                    shaped_reward[player_idx] += self.reward_shaping_params[
                        "BOILED_CHICKEN_PICKUP_REWARD"
                    ]
                elif player.get_object().name in Steakhouse_Recipe.ALL_INGREDIENTS:
                    item_type = player.get_object().name
                    if item_type != "chicken":
                        break

                    if not new_state.has_object(i_pos):
                        # Pot was empty, add boiled_chicken to it
                        new_o_id = obj_count
                        new_obj = ChickenState(new_o_id, "boiled_chicken", i_pos, [])
                        if not rollout:
                            self.object_id_dict[new_o_id] = new_obj
                        obj_count += 1
                        new_state.add_object(new_obj)
                    chicken_soup = new_state.get_object(i_pos)
                    if not chicken_soup.is_full:
                        old_soup = chicken_soup.deepcopy()
                        obj = player.remove_object()
                        chicken_soup.add_ingredient(obj)
                        chicken_soup.begin_cooking()
                        shaped_reward[player_idx] += self.reward_shaping_params[
                            "PLACEMENT_IN_POT_REW"
                        ]
                        # Log meat cooking
                        # Log potting TODO: commented for now
                        # self.log_object_potting(
                        #     events_infos,
                        #     new_state,
                        #     old_soup,
                        #     chicken_soup,
                        #     obj.name,
                        #     player_idx,
                        # # )
                        if obj.name == Steakhouse_Recipe.CHICKEN:
                            events_infos["potting_chicken"][player_idx] = True

            elif terrain_type == "G" and player.has_object():
                if (
                    player.get_object().name == "clean_plate"
                    and self.steak_ready_at_location(new_state, i_pos)
                ):
                    self.log_object_pickup(
                        events_infos, new_state, "steak", pot_states, player_idx
                    )

                    # Pick up steak
                    player.remove_object()  # Remove the clean plate
                    obj = new_state.remove_object(i_pos)  # Get steak
                    player.set_object(obj)
                    shaped_reward[player_idx] += self.reward_shaping_params[
                    "STEAK_PICKUP_REWARD"]

                elif player.get_object().name in Steakhouse_Recipe.ALL_INGREDIENTS:
                    item_type = player.get_object().name
                    if item_type != "meat":
                        break
                    if not new_state.has_object(i_pos):
                        # Pot was empty, add meat to it
                        obj = player.remove_object()
                        new_o_id = obj_count
                        new_obj = SteakState(new_o_id, "steak", i_pos, [])

                        if not rollout:
                            self.object_id_dict[new_o_id] = new_obj
                        obj_count += 1
                        new_obj.add_ingredient(obj)
                        new_obj.begin_cooking()
                        new_state.add_object(new_obj, i_pos)

                        shaped_reward[player_idx] += self.reward_shaping_params[
                            "PLACEMENT_IN_GRILL_REW"
                        ]

                        # Log meat cooking
                        events_infos["steak_cooking"][player_idx] = True

            elif terrain_type == "S" and player.has_object():
                obj = player.get_object()
                dish_name = obj.name + "_dish"
                # if (dish_name in new_state.order_list) or (dish_name in new_state._complete_orders):
                if dish_name in DISH_TYPES:
                    new_state, delivery_rew = self.deliver_dish(new_state, player, obj)
                    sparse_reward[player_idx] += delivery_rew
                    # player.num_served += 1

                    # Log dish delivery
                    events_infos["dish_delivery"][player_idx] = True

                    # If last soup necessary was delivered, stop resolving interacts
                    if (
                        new_state.order_list is not None
                        and len(new_state.order_list) == 0
                    ):
                        break

            elif terrain_type == "B":
                if player.held_object is None:
                    if new_state.has_object(i_pos):
                        obj = new_state.get_object(i_pos)
                        assert (
                            obj.name == "garnish"
                        ), "Object on chopping board was not garnish"
                        if not obj.is_ready:
                            obj.chop()
                            shaped_reward[
                                player_idx] += self.reward_shaping_params[
                                    "CHOPPING_ONION_REW"]

                            # Log onion chopping
                            events_infos["onion_chopping"][player_idx] = True

                elif player.get_object().name == "onion" and not new_state.has_object(
                    i_pos
                ):
                    # Chopping board was empty, add onion to it
                    obj = player.remove_object()
                    new_o_id = obj_count
                    new_obj = GarnishState(new_o_id, "garnish", i_pos, [])
                    if not rollout:
                        self.object_id_dict[new_o_id] = new_obj
                    obj_count += 1
                    new_obj.add_ingredient(obj)
                    new_obj.begin_chop()
                    new_state.add_object(new_obj, i_pos)
                    shaped_reward[
                    player_idx] += self.reward_shaping_params[
                    "PLACEMENT_ON_BOARD_REW"]

                    # Log onion potting
                    events_infos["onion_chopping"][player_idx] = True

                # Pick up garnish
                elif (
                    player.get_object().name == "steak"
                    and self.garnish_ready_at_location(new_state, i_pos)
                ):
                    player.remove_object()  # Remove the clean plate
                    self.log_object_pickup(
                        events_infos, new_state, "steak", pot_states, player_idx
                    )

                    _ = new_state.remove_object(i_pos)  # Get steak
                    new_o_id = obj_count
                    new_obj = IdObjectState(new_o_id, "steak_onion", pos)
                    if not rollout:
                        self.object_id_dict[new_o_id] = new_obj
                    obj_count += 1
                    player.set_object(new_obj)
                    shaped_reward[player_idx] += self.reward_shaping_params[
                        "GARNISH_STEAK_REWARD"]

                # Pick up garnish
                elif (
                    player.get_object().name == "boiled_chicken"
                    and self.garnish_ready_at_location(new_state, i_pos)
                ):
                    player.remove_object()  # Remove the clean plate
                    self.log_object_pickup(
                        events_infos,
                        new_state,
                        "boiled_chicken",
                        pot_states,
                        player_idx,
                    )

                    _ = new_state.remove_object(i_pos)  # Get steak
                    new_o_id = obj_count
                    new_obj = IdObjectState(new_o_id, "boiled_chicken_onion", pos)
                    if not rollout:
                        self.object_id_dict[new_o_id] = new_obj
                    obj_count += 1
                    player.set_object(new_obj)
                    shaped_reward[player_idx] += self.reward_shaping_params[
                        "GARNISH_STEAK_REWARD"]
            else:
                continue

            new_state.obj_count = obj_count

        return sparse_reward, shaped_reward

    def deliver_dish(self, state, player, dish_obj):
        """
        Deliver the steak, and get reward if there is no order list
        or if the type of the delivered steak matches the next order.
        """
        player.remove_object()

        if state.order_list is None:
            return state, self._delivery_reward

        # If the delivered soup is the one currently required
        # assert not self.is_terminal(state)
        current_order = state.order_list[0]
        dish = dish_obj.name + "_dish"
        if dish in current_order:
            # dish served in order
            state.order_list = state.order_list[1:]
            state._bonus_orders.append(dish + "_tick")
            state._complete_orders.append(dish + "_tick")
            state._order_display_list = state.order_list #+ state._complete_orders
            return state, self.in_order_delivery_reward
        elif dish in state.order_list:
            # dish served in not in order, but in order list
            state.order_list.remove(dish)
            state._bonus_orders.append(dish + "_tick")
            state._complete_orders.append(dish + "_tick")
            state._order_display_list = state.order_list #+ state._complete_orders
            # print("bonus orders",state._bonus_orders)
            return state, self.delivery_reward
        else:
            # dish served not in order list
            # TODO: now the dish is just lost, should log it
            state._bonus_orders.append(dish)
            # print("bonus orders",state._bonus_orders)
            return state, self.non_order_delivery_reward

    def step_environment_effects(self, state):
        state.timestep += 1

        for obj in state.objects.values():
            if obj.name == "steak":
                # automatically starts cooking when the pot has 1 ingredients
                if self.old_dynamics and (
                    not obj.is_cooking
                    and not obj.is_ready
                    and len(obj.ingredients) == 1
                ):
                    obj.begin_cooking()
                if obj.is_cooking:
                    obj.cook()
            elif obj.name == "boiled_chicken":
                # automatically starts cooking when the pot has 1 ingredients
                if (
                    not obj.is_cooking
                    and not obj.is_ready
                    and len(obj.ingredients) == 1
                ):
                    obj.begin_cooking()
                if obj.is_cooking:
                    obj.cook()

    def compute_new_positions_and_orientations(
            self, old_player_states, joint_action
        ):
            """Compute new positions and orientations ignoring collisions"""
            new_positions, new_orientations = list(
                zip(
                    *[
                        self._move_if_direction(p.position, p.orientation, a)
                        for p, a in zip(old_player_states, joint_action)
                    ]
                )
            )

            if not self.enable_same_cell:
                old_positions = tuple(p.position for p in old_player_states)
                new_positions = self._handle_collisions(old_positions, new_positions)
            return new_positions, new_orientations

    def is_terminal(self, state):
        # There is a finite horizon, handled by the environment.
        if len(state.order_list) <= 0:
            return True
        return False

    #######################
    # LAYOUT / STATE INFO #
    #######################

    def get_chopping_board_locations(self):
        return list(self.terrain_pos_dict["B"])

    def get_meat_dispenser_locations(self):
        return list(self.terrain_pos_dict["M"])
    
    def get_chicken_dispenser_locations(self):
        return list(self.terrain_pos_dict["C"])
    
    def get_sink_locations(self):
        return list(self.terrain_pos_dict["W"])

    def get_dirty_plate_locations(self):
        return list(self.terrain_pos_dict["D"])

    def get_grill_locations(self):
        return list(self.terrain_pos_dict["G"])
    
    def get_key_objects_locations(self):
        return (
            self.mdp.get_onion_dispenser_locations()
            + self.mdp.get_chopping_board_locations()
            + self.mdp.get_meat_dispenser_locations()
            + self.mdp.get_grill_locations()
            + self.mdp.get_pot_locations()
            + self.mdp.get_dirty_plate_dispenser_locations()
            + self.mdp.get_sink_locations()
        )

    def get_pot_states(self, state, pots_states_dict=None, valid_pos=None, update_knowledge_base=False):
        """Returns dict with structure:
        {
         empty: [ObjStates]
         onion: {
            'x_items': [soup objects with x items],
            'cooking': [ready soup objs]
            'ready': [ready soup objs],
            'partially_full': [all non-empty and non-full soups]
            }
         tomato: same dict structure as above
        }
        """
        if pots_states_dict is None:
            pots_states_dict = defaultdict(list)

        get_pot_info = []
        if valid_pos is not None:
            for pot_pos in self.get_pot_locations():
                if pot_pos in valid_pos:
                    get_pot_info.append(pot_pos)
        else:
            get_pot_info = self.get_pot_locations()

        for pot_pos in get_pot_info:
            if not state.has_object(pot_pos):
                pots_states_dict["empty"].append(pot_pos)
            else:
                soup = state.get_object(pot_pos)
                assert soup.name == "soup" or "chicken", (
                    "soup at "
                    + str(pot_pos)
                    + " is not a chicken/soup but a "
                    + soup.name
                )
                if soup.is_ready:
                    pots_states_dict["ready"].append(soup.id if update_knowledge_base else pot_pos)
                elif soup.is_cooking:
                    pots_states_dict["cooking"].append(soup.id if update_knowledge_base else pot_pos)
                else:
                    num_ingredients = len(soup.ingredients)
                    pots_states_dict["{}_items".format(num_ingredients)].append(pot_pos)

        return pots_states_dict

    def get_grill_states(self, state, grills_states_dict=None, valid_pos=None, update_knowledge_base=False):
        """Returns dict with structure:
        {
         empty: [positions of empty pots]
        'x_items': [grill objects with x items that have yet to start grilling],
        'cooking': [grill objs that are grilling but not ready]
        'ready': [ready grill objs],
        }
        NOTE: all returned grills are just grill positions
        """
        if grills_states_dict is None:
            grills_states_dict = defaultdict(list)

        get_grill_info = []
        if valid_pos is not None:
            for grill_pos in self.get_grill_locations():
                if grill_pos in valid_pos:
                    get_grill_info.append(grill_pos)
        else:
            get_grill_info = self.get_grill_locations()

        for grill_pos in get_grill_info:
            if not state.has_object(grill_pos):
                grills_states_dict["empty"].append(grill_pos)
            else:
                steak = state.get_object(grill_pos)
                assert steak.name == "steak", (
                    "steak at " + grill_pos + " is not a steak but a " + steak.name
                )
                if steak.is_ready:
                    grills_states_dict["ready"].append(steak.id if update_knowledge_base else grill_pos)
                else:  # steak is_cooking
                    grills_states_dict["cooking"].append(steak.id if update_knowledge_base else grill_pos)

        return grills_states_dict

    def get_ready_grills(self, grill_states):
        return grill_states["ready"]

    def get_cooking_grills(self, grill_states):
        return grill_states["cooking"]

    def get_sink_states(self, state, update_knowledge_base=False):
        empty_sink = []
        full_sink = []
        ready_sink = []
        sink_locations = self.get_sink_locations()
        for loc in sink_locations:
            if not state.has_object(loc):  # board is empty
                empty_sink.append(loc)
            else:
                obj = state.get_object(loc)
                if obj.is_ready:
                    ready_sink.append(obj.id if update_knowledge_base else loc)
                else:
                    full_sink.append(obj.id if update_knowledge_base else loc)
        return {"empty": empty_sink, "full": full_sink, "ready": ready_sink}

    def get_chopping_board_states(self, state, update_knowledge_base=False):
        empty_board = []
        full_board = []
        ready_board = []
        board_locations = self.get_chopping_board_locations()
        for loc in board_locations:
            if not state.has_object(loc):  # board is empty
                empty_board.append(loc)
            else:
                obj = state.get_object(loc)
                if obj.is_ready:
                    ready_board.append(obj.id if update_knowledge_base else loc)
                else:
                    full_board.append(obj.id if update_knowledge_base else loc)
        return {"empty": empty_board, "full": full_board, "ready": ready_board}

    def steak_ready_at_location(self, state, pos):
        if not state.has_object(pos):
            return False
        obj = state.get_object(pos)
        assert obj.name == "steak", "Object in grill was not steak"
        return obj.is_ready

    def steak_to_be_cooked_at_location(self, state, pos):
        if not state.has_object(pos):
            return False
        obj = state.get_object(pos)
        return obj.name == "steak" and not obj.is_cooking and not obj.is_ready

    def plate_clean_at_location(self, state, pos):
        if not state.has_object(pos):
            return False
        obj = state.get_object(pos)
        assert obj.name == "clean_plate", "Object in sink was not clean plate"
        return obj._cooking_tick >= obj._cook_time

    def garnish_ready_at_location(self, state, pos):
        if not state.has_object(pos):
            return False
        obj = state.get_object(pos)
        assert obj.name == "garnish", "Object on chopping board was not garnish"
        prep_time = obj._cooking_tick
        return prep_time >= obj._cook_time

    # TODO: change above objectname_ready_at_location to object_ready_at_location
    def chicken_ready_at_location(self, state, pos):
        obj_name = "boiled_chicken"
        if not state.has_object(pos):
            return False
        obj = state.get_object(pos)
        assert obj.name == obj_name, "Object at location was not {}".format(obj_name)
        return obj.is_ready
    
    def _check_valid_state(self, state):
        """Checks that the state is valid.

        Conditions checked:
        - Players are on free spaces, not terrain
        - Held objects have the same position as the player holding them
        - Non-held objects are on terrain
        - No two players or non-held objects occupy the same position
        - Objects have a valid state (eg. no pot with 4 onions)
        """
        all_objects = list(state.objects.values())
        for player_state in state.players:
            # Check that players are not on terrain
            pos = player_state.position
            assert pos in self.get_valid_player_positions()

            # Check that held objects have the same position
            if player_state.held_object is not None:
                all_objects.append(player_state.held_object)
                assert (
                    player_state.held_object.position == player_state.position
                )

        for obj_pos, obj_state in state.objects.items():
            # Check that the hash key position agrees with the position stored
            # in the object state
            assert obj_state.position == obj_pos
            # Check that non-held objects are on terrain
            assert self.get_terrain_type_at_pos(obj_pos) != " "

        # Check that players and non-held objects don't overlap
        if not self.enable_same_cell:
            all_pos = [player_state.position for player_state in state.players]
            all_pos += [obj_state.position for obj_state in state.objects.values()]
            assert len(all_pos) == len(
                set(all_pos)
            ), "Overlapping players or objects"

        # Check that objects have a valid state
        for obj_state in all_objects:
            assert obj_state.is_valid()

    ################################
    # EVENT LOGGING HELPER METHODS #
    ################################

    def log_object_drop(self, events_infos, state, obj_name, pot_states, player_index):
        """Player dropped the object on a counter"""
        obj_drop_key = obj_name + "_drop"
        if obj_drop_key not in events_infos:
            # TODO: add support for tomato event logging
            if obj_name in [
                "meat",
                "clean_plate",
                "steak",
                "garnish",
                "chicken",
                "boiled_chicken",
            ]:
                return
            raise ValueError("Unknown event {}".format(obj_drop_key))

    def is_potting_optimal(self, state, old_soup, new_soup):
        """
        True if the highest valued soup possible is the same before and after the potting
        """
        old_recipe = (
            Steakhouse_Recipe(old_soup.ingredients) if old_soup.ingredients else None
        )
        new_recipe = Steakhouse_Recipe(new_soup.ingredients)
        old_val = self.get_recipe_value(
            state, self.get_optimal_possible_recipe(state, old_recipe)
        )
        new_val = self.get_recipe_value(
            state, self.get_optimal_possible_recipe(state, new_recipe)
        )
        return old_val == new_val

    def is_potting_viable(self, state, old_soup, new_soup):
        """
        True if there exists a non-zero reward soup possible from new ingredients
        """
        new_recipe = Steakhouse_Recipe(new_soup.ingredients)
        new_val = self.get_recipe_value(
            state, self.get_optimal_possible_recipe(state, new_recipe)
        )
        return new_val > 0

    def is_potting_catastrophic(self, state, old_soup, new_soup):
        """
        True if no non-zero reward soup is possible from new ingredients
        """
        old_recipe = (
            Steakhouse_Recipe(old_soup.ingredients) if old_soup.ingredients else None
        )
        new_recipe = Steakhouse_Recipe(new_soup.ingredients)
        old_val = self.get_recipe_value(
            state, self.get_optimal_possible_recipe(state, old_recipe)
        )
        new_val = self.get_recipe_value(
            state, self.get_optimal_possible_recipe(state, new_recipe)
        )
        return old_val > 0 and new_val == 0

    def is_potting_useless(self, state, old_soup, new_soup):
        """
        True if ingredient added to a soup that was already gauranteed to be worth at most 0 points
        """
        old_recipe = (
            Steakhouse_Recipe(old_soup.ingredients) if old_soup.ingredients else None
        )
        old_val = self.get_recipe_value(
            state, self.get_optimal_possible_recipe(state, old_recipe)
        )
        return old_val == 0

    #####################
    # TERMINAL GRAPHICS #
    #####################

    def state_string(self, state):
        """String representation of the current state"""
        # TODO
        return ""

    ###################
    # STATE ENCODINGS #
    ###################

    def get_lossless_state_encoding_shape(self):
        return np.array(list(self.shape) + [48])
    
    def lossless_state_encoding(
        self, steakhouse_state: SteakhouseState, horizon: int = 400
    ):
        """
        Creidts to Saeed Hedayatian (hedayatians@gmail.com).
        Featurizes a SteakhouseState object into a stack of boolean masks that are
        easily readable by a CNN

        Args:
            steakhouse_state: The current state.
            horizon: Time horizon of an episode. (default: 400)

        Returns:
            State encoded in the form of a (w, h, 48) grid for each player, with the
                following features as channels:
                - Cur player location (1)
                - Other player location (1)
                - Cur player direction (one-hot; 4)
                - Other player direction (one-hot; 4)
                - Base map features (locations of various map elements; 11)
                - Variable map features (locations of movable objects, orders being
                    cooked, etc.; 22)
                - Urgency feature (true if <40 timesteps to completion; 1)
                - Current order feature (one-hot; 4)
        """
        assert (
            self.num_players == 2
        ), "Functionality has to be added to support encondings for > 2 players"
        base_map_features = [
            "counter_loc",
            "pot_loc",
            "dirty_plate_disp_loc",
            "onion_disp_loc",
            "serve_loc",
            "grill_loc",
            "chicken_disp_loc",
            "sink_loc",
            "meat_disp_loc",
            "chopping_board_loc",
        ]
        variable_map_features = [
            "onions",
            "chickens",
            "meats",
            "dirty_plates",
            "steak_onions",
            "boiled_chicken_onions",
            "chicken_cook_time_remaining",
            "chicken_done",
            "steak_cook_time_remaining",
            "steak_done",
            "plate_clean_time_remaining",
            "plate_cleaned",
            "garnish_chop_time_remaining",
            "garnish_chopped",
        ]
        urgency_features = ["urgency"]
        all_objects = steakhouse_state.all_objects_list

        def make_layer(position, value):
            layer = np.zeros(self.shape)
            layer[position] = value
            return layer

        def process_for_player(primary_agent_idx):
            # Ensure that primary_agent_idx layers are ordered before other_agent_idx
            # layers
            other_agent_idx = 1 - primary_agent_idx
            ordered_player_features = [
                f"player_{primary_agent_idx}_loc",
                f"player_{other_agent_idx}_loc",
            ] + [
                f"player_{i}_orientation_{Direction.DIRECTION_TO_INDEX[d]}"
                for i, d in itertools.product(
                    [primary_agent_idx, other_agent_idx],
                    Direction.ALL_DIRECTIONS,
                )
            ]

            LAYERS = (
                ordered_player_features
                + base_map_features
                + variable_map_features
                + urgency_features
                + DISH_TYPES
            )
            state_mask_dict = {k: np.zeros(self.shape) for k in LAYERS}

            # MAP LAYERS
            if horizon - steakhouse_state.timestep < 40:
                state_mask_dict["urgency"] = np.ones(self.shape)

            for loc in self.get_counter_locations():
                state_mask_dict["counter_loc"][loc] = 1

            for loc in self.get_pot_locations():
                state_mask_dict["pot_loc"][loc] = 1

            for loc in self.get_dirty_plate_locations():
                state_mask_dict["dirty_plate_disp_loc"][loc] = 1

            for loc in self.get_onion_dispenser_locations():
                state_mask_dict["onion_disp_loc"][loc] = 1

            for loc in self.get_serving_locations():
                state_mask_dict["serve_loc"][loc] = 1

            for loc in self.get_grill_locations():
                state_mask_dict["grill_loc"][loc] = 1

            for loc in self.get_chicken_dispenser_locations():
                state_mask_dict["chicken_disp_loc"][loc] = 1

            for loc in self.get_sink_locations():
                state_mask_dict["sink_loc"][loc] = 1

            for loc in self.get_meat_dispenser_locations():
                state_mask_dict["meat_disp_loc"][loc] = 1

            for loc in self.get_chopping_board_locations():
                state_mask_dict["chopping_board_loc"][loc] = 1

            # Current order layers
            if steakhouse_state.order_list:
                # Order list is not None and there is at least one order remaining
                cur_order = steakhouse_state.order_list[0]
                state_mask_dict[cur_order] = np.ones(self.shape)

            # PLAYER LAYERS
            for i, player in enumerate(steakhouse_state.players):
                player_orientation_idx = Direction.DIRECTION_TO_INDEX[
                    player.orientation
                ]
                state_mask_dict[f"player_{i}_loc"] = make_layer(player.position, 1)
                state_mask_dict[f"player_{i}_orientation_{player_orientation_idx}"] = (
                    make_layer(player.position, 1)
                )

            # OBJECT & STATE LAYERS
            for obj in all_objects:
                if obj.name == "boiled_chicken":
                    # Boiled chicken is similar to soup except that it immediately
                    # starts cooking and only needs 1 chicken.
                    if obj.position in self.get_pot_locations():
                        # Only one chicken can be in pot and it is never idle. When
                        # player interacts with pot holding chicken, a `ChickenState` is
                        # created, chicken is added, and cooking starts.
                        state_mask_dict["chicken_cook_time_remaining"] += make_layer(
                            obj.position, obj.cook_time - obj._cooking_tick
                        )
                        if obj.is_ready:
                            state_mask_dict["chicken_done"] += make_layer(
                                obj.position, 1
                            )
                    else:
                        # If boiled chicken is not in a pot, treat it like a soup that
                        # is cooked with remaining time 0
                        state_mask_dict["chicken_done"] += make_layer(obj.position, 1)

                elif obj.name == "steak":
                    # Steak is similar to boiled chicken.
                    if obj.position in self.get_grill_locations():
                        state_mask_dict["steak_cook_time_remaining"] += make_layer(
                            obj.position, obj.cook_time - obj._cooking_tick
                        )
                        if obj.is_ready:
                            state_mask_dict["steak_done"] += make_layer(obj.position, 1)
                    else:
                        state_mask_dict["steak_done"] += make_layer(obj.position, 1)

                elif obj.name == "clean_plate":
                    # Cleaning plate is similar to steak and boiled chicken for
                    # observation (but interact action is required to move the cleaning
                    # forward unlike cooking which happens automatically).
                    if obj.position in self.get_sink_locations():
                        state_mask_dict["plate_clean_time_remaining"] += make_layer(
                            obj.position, obj.cook_time - obj._cooking_tick
                        )
                        if obj.is_ready:
                            state_mask_dict["plate_cleaned"] += make_layer(
                                obj.position, 1
                            )
                    else:
                        state_mask_dict["plate_cleaned"] += make_layer(obj.position, 1)

                elif obj.name == "garnish":
                    # Cutting for garnish is similar to cleaning plate.
                    if obj.position in self.get_chopping_board_locations():
                        state_mask_dict["garnish_chop_time_remaining"] += make_layer(
                            obj.position, obj.cook_time - obj._cooking_tick
                        )
                        if obj.is_ready:
                            state_mask_dict["garnish_chopped"] += make_layer(
                                obj.position, 1
                            )
                    else:
                        state_mask_dict["garnish_chopped"] += make_layer(
                            obj.position, 1
                        )

                elif obj.name == "onion":
                    state_mask_dict["onions"] += make_layer(obj.position, 1)
                elif obj.name == "chicken":
                    state_mask_dict["chickens"] += make_layer(obj.position, 1)
                elif obj.name == "meat":
                    state_mask_dict["meats"] += make_layer(obj.position, 1)
                elif obj.name == "dirty_plate":
                    state_mask_dict["dirty_plates"] += make_layer(obj.position, 1)
                elif obj.name == "steak_onion":
                    # Garnished steak doesn't need cooking, so treated as a regular
                    # object.
                    state_mask_dict["steak_onions"] += make_layer(obj.position, 1)
                elif obj.name == "boiled_chicken_onion":
                    # Garnished chicken doesn't need cooking, so treated as a regular
                    # object.
                    state_mask_dict["boiled_chicken_onions"] += make_layer(
                        obj.position, 1
                    )
                else:
                    raise ValueError("Unrecognized object")

            logger.debug("terrain----")
            logger.debug(np.array(self.terrain_mtx))
            logger.debug("-----------")
            logger.debug(len(LAYERS))
            logger.debug(len(state_mask_dict))
            for k, v in state_mask_dict.items():
                logger.debug(k)
                logger.debug(np.transpose(v, (1, 0)))

            # Stack of all the state masks, order decided by order of LAYERS
            state_mask_stack = np.array(
                [state_mask_dict[layer_id] for layer_id in LAYERS]
            )
            state_mask_stack = np.transpose(state_mask_stack, (1, 2, 0))
            assert state_mask_stack.shape[:2] == self.shape
            assert state_mask_stack.shape[2] == len(LAYERS)
            # NOTE: currently not including time left or order_list in featurization
            return np.array(state_mask_stack).astype(int)

        # NOTE: Currently not very efficient, a decent amount of computation repeated
        # here
        num_players = len(steakhouse_state.players)
        final_obs_for_players = tuple(process_for_player(i) for i in range(num_players))
        return final_obs_for_players


    def get_featurize_state_shape(self, num_pots=2):
        pass
        # TODO
    
    def _normalize_pos(self, pos_or):
        """Normalizes positions and orientations to be in [-1, 1]"""
        pos_or = np.array(pos_or).reshape((-1,))
        return np.array(
            [
                2 * pos_or[0] / self.width - 1,
                2 * pos_or[1] / self.height - 1,
                pos_or[2],
                pos_or[3],
            ]
        )

    def _get_object_held_feature_for_player(
        self, state: SteakhouseState, player_index: int
    ):
        all_object_names = [
            "meat",
            "chicken",
            "dirty_plate",
            "clean_plate",
            "onion",
            "steak",
            "boiled_chicken",
            "steak_onion",
            "boiled_chicken_onion",
        ]
        objs = np.zeros((len(all_object_names),))
        if state.players[player_index].held_object is None:
            return objs
        for object_index, object_name in enumerate(all_object_names):
            if state.players[player_index].held_object.name.lower() == object_name:
                objs[object_index] = 1
                return objs
        else:
            raise ValueError(
                f"Unrecognized object held by player {player_index}: {state.players[player_index].held_object.name}"
            )

    def get_vector_feature_size(self, num_important_counters: int = 0):
        return (
            self.num_players * (4 + 9)  # 4 -> pos_or, 9 -> num of objects
            + len(self.get_pot_locations())
            + len(self.get_grill_locations())
            + len(self.get_sink_locations())
            + len(self.get_chopping_board_locations())
            + num_important_counters * 9  # 9 -> num of objects
            + 4 * 2  # number of possible dishes * number of future orders we see
        )

    def get_vector_feature(
        self, state: SteakhouseState, important_counters: List = None
    ) -> Tuple[np.ndarray]:
        """Encodes the state into a 1d vector using handcrafted features.
        The encoding contains:
            1. Position and orientation of each player
            2. Objects held by each agent
            3. Status of pots
            4. Status of grills
            5. Status of sinks
            6. Status of chopping boards
            7. Status of selected (important) counters
            8. Type of next 2 orders
        Args:
            state: Full game state that is to be transformed
            important_counters: An optional list of counters whose information should be added to the feature vector. Used primarily by the hierarchical agent.
        Returns:
            A tuple containing the feature vectors of each player.
        """
        if important_counters is None:
            important_counters = []
        player_features = []
        for index in range(self.num_players):
            current_player_features = []

            # position and orientations + objects held
            pos_and_or = state.players_pos_and_or
            current_player_features.append(self._normalize_pos(pos_and_or[index]))
            current_player_features.append(
                self._get_object_held_feature_for_player(state, index)
            )
            for i in range(self.num_players):
                if i == index:
                    continue
                current_player_features.append(self._normalize_pos(pos_and_or[i]))
                current_player_features.append(
                    self._get_object_held_feature_for_player(state, i)
                )

            player_features.append(current_player_features)

        shared_features = []
        # Pots
        pot_states = np.zeros((len(self.get_pot_locations()),))
        for i, pot_pos in enumerate(self.get_pot_locations()):
            if not state.has_object(pot_pos):
                pot_states[i] = -1  # empty pot
            else:
                soup = state.get_object(pot_pos)
                assert soup.name == "soup" or "chicken", (
                    "soup at "
                    + str(pot_pos)
                    + " is not a chicken/soup but a "
                    + soup.name
                )
                pot_states[i] = max(soup._cooking_tick, 0) / soup.cook_time
                if soup.is_ready:
                    assert pot_states[i] == 1  # ready pot
                elif soup.is_cooking:
                    assert 0 <= pot_states[i] < 1  # cooking pot
                else:
                    # This shouldn't be triggered:? unless we add onion soups or other recipies that require multiple ingredients
                    raise ValueError(
                        f"Unexpected soup in the pot. name: {soup.name}, ingredients: {soup.ingredients}"
                    )
        shared_features.append(pot_states)

        # Grills
        grill_states = np.zeros((len(self.get_grill_locations()),))
        for i, grill_pos in enumerate(self.get_grill_locations()):
            if not state.has_object(grill_pos):
                grill_states[i] = -1  # Empty grill
            else:
                steak = state.get_object(grill_pos)
                assert steak.name == "steak", (
                    "steak at " + grill_pos + " is not a steak but a " + steak.name
                )
                grill_states[i] = max(steak._cooking_tick, 0) / steak.cook_time
                if steak.is_ready:
                    assert grill_states[i] == 1  # Ready steak
                else:
                    assert 0 <= grill_states[i] < 1  # Cooking steak
        shared_features.append(grill_states)

        # Sinks
        sink_states = np.zeros((len(self.get_sink_locations()),))
        for i, sink_pos in enumerate(self.get_sink_locations()):
            if not state.has_object(sink_pos):  # Empty sink
                sink_states[i] = -1
            else:
                obj = state.get_object(sink_pos)
                sink_states[i] = max(obj._cooking_tick, 0) / obj.cook_time
                if obj.is_ready:
                    assert sink_states[i] == 1  # Cleaned dish in sink
                else:
                    assert 0 <= sink_states[i] < 1  # Half-cleaned dish in sink
        shared_features.append(sink_states)

        # Chopping Boards
        board_states = np.zeros((len(self.get_chopping_board_locations()),))
        for i, board_pos in enumerate(self.get_chopping_board_locations()):
            if not state.has_object(board_pos):  # Empty board
                board_states[i] = 0
            else:
                obj = state.get_object(board_pos)
                board_states[i] = max(obj._cooking_tick, 0) / obj.cook_time
                if obj.is_ready:
                    assert board_states[i] == 1  # Chopped onion on board
                else:
                    assert 0 <= board_states[i] < 1  # Half-chopped onion on board
        shared_features.append(board_states)

        # Counters
        all_object_names = [
            "meat",
            "chicken",
            "dirty_plate",
            "clean_plate",
            "onion",
            "steak",
            "boiled_chicken",
            "steak_onion",
            "boiled_chicken_onion",
        ]
        if len(important_counters) > 0:
            counter_states = np.zeros((len(important_counters), len(all_object_names)))
            for i, counter_pos in enumerate(important_counters):
                if not state.has_object(counter_pos):
                    continue  # Empty
                else:
                    obj = state.get_object(counter_pos)
                    for object_index, object_name in enumerate(all_object_names):
                        if obj.name.lower() == object_name:
                            counter_states[i, object_index] = 1
                            break
                    else:
                        raise ValueError(
                            f"Unrecognized object found at {counter_pos}: {obj.name}"
                        )
            shared_features.append(counter_states.reshape((-1,)))

        # Next orders
        _order_index = {
            "steak_dish": 0,
            "boiled_chicken_dish": 1,
            "steak_onion_dish": 2,
            "boiled_chicken_onion_dish": 3,
        }
        _num_dishes, _horizon = len(_order_index.keys()), 2
        next_orders = np.zeros((_num_dishes * _horizon,))
        for i, order in enumerate(state.order_list[:_horizon]):
            next_orders[i * _num_dishes + _order_index[order]] = 1
        shared_features.append(next_orders)

        player_features = [
            np.concatenate(pf + shared_features) for pf in player_features
        ]

        return player_features

    ##############
    # DEPRECATED #
    ##############

    ###################
    # RENDER FUNCTION #
    ###################