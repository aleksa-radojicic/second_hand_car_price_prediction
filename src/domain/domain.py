from sqlalchemy import Column, String
from sqlalchemy.orm import mapped_column, relationship
from sqlalchemy.orm.decl_api import declarative_base
from sqlalchemy.sql.schema import ForeignKey
from sqlalchemy.sql.sqltypes import BigInteger, Text

Base = declarative_base()
DEF_STR_SIZE = 255
DSS = DEF_STR_SIZE  # Alias


class Listing(Base):
    __table_args__ = {"extend_existing": True, "comment": "Oglas"}
    __tablename__ = "listings"

    id = mapped_column(BigInteger, primary_key=True, comment="Broj oglasa")
    name = Column(String(DSS))
    short_url = Column(String(2 * DSS))
    price = Column(String(DSS))
    listing_followers_no = Column(String(DSS))
    location = Column(String(DSS))
    images_no = Column(String(DSS))

    safety = Column(Text, comment="Sigurnost")
    equipment = Column(Text, comment="Oprema")
    other = Column(Text, comment="Stanje")
    description = Column(Text, comment="Opis")

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

    SRB_SECTION_NAMES_TO_ATTRS_MAP = {
        safety.comment: "safety",
        equipment.comment: "equipment",
        other.comment: "other",
        description.comment: "description",
    }

    def __repr__(self):
        kv_dict = todict(self)
        excl = (
            "_sa_adapter",
            "_sa_instance_state",
            "general_information",
            "additional_information",
            "safety",
            "equipment",
            "other",
            "description",
        )
        for k in kv_dict.copy():
            if k in excl:
                kv_dict.pop(k)

        params = ", ".join(f"{k}={v}" for k, v in kv_dict.items())
        return f"{self.__class__.__name__}({params})"


def todict(obj):
    return {k: v for k, v in vars(obj).items() if not k.startswith("_")}


class GeneralInformation(Base):
    __table_args__ = {"extend_existing": True, "comment": "Opšte informacije"}
    __tablename__ = "general_informations"

    id = mapped_column(
        ForeignKey("listings.id", ondelete="CASCADE", onupdate="CASCADE"),
        primary_key=True,
    )
    listing = relationship(Listing, uselist=False, back_populates="general_information")

    condition = Column(String(DSS), comment="Stanje")
    brand = Column(String(DSS), comment="Marka")
    model = Column(String(DSS), comment="Model")
    production_year = Column(String(DSS), comment="Godište")
    kilometerage = Column(String(DSS), comment="Kilometraža")
    body_type = Column(String(DSS), comment="Karoserija")
    fuel_type = Column(String(DSS), comment="Gorivo")
    engine_capacity = Column(String(DSS), comment="Kubikaža")
    engine_power = Column(String(DSS), comment="Snaga motora")
    fixed_price = Column(String(DSS), comment="Fiksna cena")
    trade_in = Column(String(DSS), comment="Zamena")
    certified = Column(String(DSS), comment="Atestiran")
    battery_capacity = Column(String(DSS), comment="Kapacitet baterije")

    SRB_NAMES_TO_ATTRS_MAP = {
        condition.comment: "condition",
        brand.comment: "brand",
        model.comment: "model",
        production_year.comment: "production_year",
        kilometerage.comment: "kilometerage",
        body_type.comment: "body_type",
        fuel_type.comment: "fuel_type",
        engine_capacity.comment: "engine_capacity",
        engine_power.comment: "engine_power",
        fixed_price.comment: "fixed price",
        trade_in.comment: "trade_in",
        certified.comment: "certified",
        battery_capacity.comment: "battery_capacity",
    }


class AdditionalInformation(Base):
    __table_args__ = {"extend_existing": True, "comment": "Dodatne informacije"}
    __tablename__ = "additional_informations"

    id = mapped_column(
        ForeignKey("listings.id", ondelete="CASCADE", onupdate="CASCADE"),
        primary_key=True,
    )
    listing = relationship(
        Listing, uselist=False, back_populates="additional_information"
    )

    floating_flywheel = Column(String(DSS), comment="Plivajući zamajac")
    engine_emission_class = Column(String(DSS), comment="Emisiona klasa motora")
    propulsion = Column(String(DSS), comment="Pogon")
    gearbox_type = Column(String(DSS), comment="Menjač")
    doors_no = Column(String(DSS), comment="Broj vrata")
    seats_no = Column(String(DSS), comment="Broj sedišta")
    steering_wheel_side = Column(String(DSS), comment="Strana volana")
    air_conditioning = Column(String(DSS), comment="Klima")
    color = Column(String(DSS), comment="Boja")
    interior_material = Column(String(DSS), comment="Materijal enterijera")
    interior_color = Column(String(DSS), comment="Boja enterijera")
    registered_until = Column(String(DSS), comment="Registrovan do")
    vehicle_origin = Column(String(DSS), comment="Poreklo vozila")
    ownership = Column(String(DSS), comment="Vlasništvo")
    damage = Column(String(DSS), comment="Oštećenje")
    import_country = Column(String(DSS), comment="Zemlja uvoza")
    sales_method = Column(String(DSS), comment="Način prodaje")
    credit = Column(String(DSS), comment="Kredit")
    deposit = Column(String(DSS), comment="Učešće (depozit)")
    installment_no = Column(String(DSS), comment="Broj rata")
    installment_amount = Column(String(DSS), comment="Visina rate")
    interest_free_credit = Column(String(DSS), comment="Beskamatni kredit")
    leasing = Column(String(DSS), comment="Lizing")
    cash_payment = Column(String(DSS), comment="Gotovinska uplata")
    range_on_full_battery_km = Column(
        String(DSS), comment="Domet sa punom baterijom (km)"
    )

    SRB_NAMES_TO_ATTRS_MAP = {
        floating_flywheel.comment: "floating_flywheel",
        engine_emission_class.comment: "engine_emission_class",
        propulsion.comment: "propulsion",
        gearbox_type.comment: "gearbox_type",
        doors_no.comment: "doors_no",
        seats_no.comment: "seats_no",
        steering_wheel_side.comment: "steering_wheel_side",
        air_conditioning.comment: "air_conditioning",
        color.comment: "color",
        interior_material.comment: "interior_material",
        interior_color.comment: "interior_color",
        registered_until.comment: "registered_until",
        vehicle_origin.comment: "vehicle_origin",
        ownership.comment: "ownership",
        damage.comment: "damage",
        import_country.comment: "import_country",
        sales_method.comment: "sales_method",
        credit.comment: "credit",
        deposit.comment: "deposit",
        installment_no.comment: "installment_no",
        installment_amount.comment: "installment_amount",
        interest_free_credit.comment: "interest_free_credit",
        leasing.comment: "leasing",
        cash_payment.comment: "cash_payment",
        range_on_full_battery_km.comment: "range_on_full_battery_km",
    }
