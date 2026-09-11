"""Official rigid FlyGym attachment with the same flat-ground contacts."""

from flygym.anatomy import BaseContactBodiesPreset
from flygym.compose import FlatGroundWorld
from flygym.compose.physics import ContactParams
from flygym.compose.world.tethered_world import TetheredWorld


class TetheredGroundWorld(FlatGroundWorld):
    def _attach_fly_mjcf(
        self,
        fly,
        spawn_position,
        spawn_rotation,
        *,
        bodysegs_with_ground_contact,
        add_ground_contact_sensors=False,
        ground_contact_params=None,
    ):
        segments = bodysegs_with_ground_contact
        preset_cls = type(fly).CONTACT_BODIES_PRESET_CLASS
        if isinstance(segments, str):
            segments = preset_cls(segments)
        if isinstance(segments, BaseContactBodiesPreset):
            if not isinstance(segments, preset_cls):
                raise TypeError("Contact preset does not match fly model")
            segments = segments.to_body_segments_list()
        result = TetheredWorld._attach_fly_mjcf(
            self, fly, spawn_position, spawn_rotation
        )
        self._set_ground_contact(
            fly, segments, ground_contact_params or ContactParams()
        )
        if add_ground_contact_sensors:
            self._add_ground_contact_sensors(fly, segments)
        return result
