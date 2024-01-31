from sqlalchemy import Column, String
from sqlalchemy.orm import mapped_column, relationship
from sqlalchemy.orm.decl_api import declarative_base
from sqlalchemy.sql.schema import ForeignKey
from sqlalchemy.sql.sqltypes import BigInteger

Base = declarative_base()
DEFAULT_STRING_SIZE = 255
section_names_srb = [
    "Opšte informacije",
    "Dodatne informacije",
    "Sigurnost",
    "Oprema",
    "Stanje",
]


class Listing(Base):
    __table_args__ = {"extend_existing": True}
    __tablename__ = "listings"

    srb_names_to_attrs_map = {
        "Broj oglasa": "id",
        **dict(
            zip(
                section_names_srb,
                [
                    "general_information",
                    "additional_information",
                    "safety_information",
                    "equipment_information",
                    "other_information",
                ],
            )
        ),
    }

    id = mapped_column(BigInteger, primary_key=True)
    name = Column(String(DEFAULT_STRING_SIZE))
    short_url = Column(String(2 * DEFAULT_STRING_SIZE), default=f"{id}/{name}")
    price = Column(String(DEFAULT_STRING_SIZE))
    listing_followers_no = Column(String(DEFAULT_STRING_SIZE))
    location = Column(String(DEFAULT_STRING_SIZE))
    images_no = Column(String(DEFAULT_STRING_SIZE))

    general_information = relationship(
        "GeneralInformation",
        uselist=False,
        back_populates="listing",
        cascade="all, delete-orphan",
    )
    additional_information = relationship(
        "AdditionalInformation",
        uselist=False,
        back_populates="listing",
        cascade="all, delete-orphan",
    )
    safety_information = relationship(
        "SafetyInformation",
        uselist=False,
        back_populates="listing",
        cascade="all, delete-orphan",
    )
    equipment_information = relationship(
        "EquipmentInformation",
        uselist=False,
        back_populates="listing",
        cascade="all, delete-orphan",
    )
    other_information = relationship(
        "OtherInformation",
        uselist=False,
        back_populates="listing",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        params = ", ".join(f"{k}={v}" for k, v in todict(self).items())
        return f"{self.__class__.__name__}({params})"


def todict(obj):
    excl = ("_sa_adapter", "_sa_instance_state")
    return {
        k: v
        for k, v in vars(obj).items()
        if not k.startswith("_") and not any(hasattr(v, a) for a in excl)
    }


class GeneralInformation(Base):
    __table_args__ = {"extend_existing": True}
    __tablename__ = "general_informations"

    srb_names_to_attrs_map = {
        "Stanje": "condition",
        "Marka": "brand",
        "Model": "model",
        "Kilometraža": "production_year",
        "Karoserija": "body_type",
        "Gorivo": "fuel_type",
        "Kubikaža": "engine_capacity",
        "Snaga motora": "engine_power",
        "Fiksna cena": "fixed price",
        "Zamena": "trade_in",
    }

    id = mapped_column(
        ForeignKey("listings.id", ondelete="CASCADE", onupdate="RESTRICT"),
        primary_key=True,
    )
    listing = relationship(Listing, uselist=False, back_populates="general_information")

    condition = Column(String(DEFAULT_STRING_SIZE))  # Stanje
    brand = Column(String(DEFAULT_STRING_SIZE))  # Marka
    model = Column(String(DEFAULT_STRING_SIZE))  # Model
    production_year = Column(String(DEFAULT_STRING_SIZE))  # Kilometraža
    body_type = Column(String(DEFAULT_STRING_SIZE))  # Karoserija
    fuel_type = Column(String(DEFAULT_STRING_SIZE))  # Gorivo
    engine_capacity = Column(String(DEFAULT_STRING_SIZE))  # Kubikaža
    engine_power = Column(String(DEFAULT_STRING_SIZE))  # Snaga motora
    fixed_price = Column(String(DEFAULT_STRING_SIZE))  # Fiksna cena
    trade_in = Column(String(DEFAULT_STRING_SIZE))  # Zamena


class AdditionalInformation(Base):
    __table_args__ = {"extend_existing": True}
    __tablename__ = "additional_informations"

    srb_names_to_attrs_map = {
        "Plivajući zamajac": "floating_flywheel",
        "Emisiona klasa motora": "engine_emission_class",
        "Pogon": "propulsion",
        "Menjač": "gearbox_type",
        "Broj vrata": "doors_no",
        "Broj sedišta": "seats_no",
        "Strana volana": "steering_wheel_side",
        "Klima": "air_conditioning",
        "Kredit": "credit",
        "Zamena": "trade_in",
        "Boja": "color",
        "Materijal enterijera": "interior_material",
        "Boja enterijera": "interior_color",
        "Registrovan do": "registered_until",
        "Poreklo vozila": "vehicle_origin",
        "Vlasništvo": "ownership",
        "Oštećenje": "damage",
        "Zemlja uvoza": "import_country",
    }

    id = mapped_column(
        ForeignKey("listings.id", ondelete="CASCADE", onupdate="RESTRICT"),
        primary_key=True,
    )
    listing = relationship(
        Listing, uselist=False, back_populates="additional_information"
    )

    floating_flywheel = Column(String(DEFAULT_STRING_SIZE))  # Plivajući zamajac
    engine_emission_class = Column(String(DEFAULT_STRING_SIZE))  # Emisiona klasa motora
    propulsion = Column(String(DEFAULT_STRING_SIZE))  # Pogon
    gearbox_type = Column(String(DEFAULT_STRING_SIZE))  # Menjač
    doors_no = Column(String(DEFAULT_STRING_SIZE))  # Broj vrata
    seats_no = Column(String(DEFAULT_STRING_SIZE))  # Broj sedišta
    steering_wheel_side = Column(String(DEFAULT_STRING_SIZE))  # Strana volana
    air_conditioning = Column(String(DEFAULT_STRING_SIZE))  # Klima
    credit = Column(String(DEFAULT_STRING_SIZE))  # Kredit
    color = Column(String(DEFAULT_STRING_SIZE))  # Boja
    interior_material = Column(String(DEFAULT_STRING_SIZE))  # Materijal enterijera
    interior_color = Column(String(DEFAULT_STRING_SIZE))  # Boja enterijera
    registered_until = Column(String(DEFAULT_STRING_SIZE))  # Registrovan do
    vehicle_origin = Column(String(DEFAULT_STRING_SIZE))  # Poreklo vozila
    ownership = Column(String(DEFAULT_STRING_SIZE))  # Vlasništvo
    damage = Column(String(DEFAULT_STRING_SIZE))  # Oštećenje
    import_country = Column(String(DEFAULT_STRING_SIZE))  # Zemlja uvoza


class SafetyInformation(Base):
    __table_args__ = {"extend_existing": True}
    __tablename__ = "safety_informations"

    srb_names_to_attrs_map = {"Sigurnost": "safety"}

    id = mapped_column(
        ForeignKey("listings.id", ondelete="CASCADE", onupdate="RESTRICT"),
        primary_key=True,
    )
    listing = relationship(Listing, uselist=False, back_populates="safety_information")

    safety = Column(String(10_000))


class EquipmentInformation(Base):
    __table_args__ = {"extend_existing": True}
    __tablename__ = "safety_informations"

    srb_names_to_attrs_map = {"Oprema": "equipment"}

    id = mapped_column(
        ForeignKey("listings.id", ondelete="CASCADE", onupdate="RESTRICT"),
        primary_key=True,
    )
    listing = relationship(
        Listing, uselist=False, back_populates="equipment_information"
    )

    equipment = Column(String(10_000))


class OtherInformation(Base):
    __table_args__ = {"extend_existing": True}
    __tablename__ = "safety_informations"

    srb_names_to_attrs_map = {"Stanje": "other"}

    id = mapped_column(
        ForeignKey("listings.id", ondelete="CASCADE", onupdate="RESTRICT"),
        primary_key=True,
    )
    listing = relationship(Listing, uselist=False, back_populates="other_information")

    other = Column(String(10_000))
