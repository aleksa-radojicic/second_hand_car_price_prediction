from dataclasses import dataclass

from src.features.utils import CustomTransformer
from src.utils import Dataset, Metadata, preprocess_init


@dataclass
class MACleanerConfig:
    finalize_flag: bool = True


class MACleaner(CustomTransformer):
    finalize_flag: bool

    def __init__(self, finalize_flag: bool = True):
        super().__init__()
        self.finalize_flag = finalize_flag

    @staticmethod
    @preprocess_init
    def ma_finalize(df: Dataset, metadata: Metadata) -> tuple[Dataset, Metadata]:
        features_info = metadata.features_info
        features_info["features_to_delete"].remove("gi_battery_capacity")
        features_info["features_to_delete"].remove("ai_range_on_full_battery_km")

        low_importance_features = [
            "ai_credit",
            "ai_interest_free_credit",
            "ai_leasing",
            "e_Prednja_noćna_kamera",
            "e_Keramičke_kočnice",
            "e_Torba_za_skije",
            "e_Autonomna_vožnja",
            "e_Postolje_za_bežično_punjenje_telefona",
            "e_Ventilacija_sedišta",
            "e_Matrix_farovi",
            "e_Android_Auto",
            "e_Grejanje_volana",
            "e_Apple_CarPlay",
            "e_Privlačenje_vrata_pri_zatvaranju",
            "e_Head_up_display",
            "e_Masažna_sedišta",
            "gi_battery_capacity",
            "ai_range_on_full_battery_km",
            "e_Automatsko_parkiranje",
            "e_Webasto",
            "e_Memorija_sedišta",
            "e_Otvor_za_skije",
            "e_Retrovizor_se_obara_pri_rikvercu",
            "e_Zavesice_na_zadnjim_prozorima",
            "e_Šiber",
            "e_Subwoofer",
            "e_Sportska_sedišta",
            "e_Upravljanje_na_sva_četiri_točka",
            "e_DVD_ili_TV",
            "e_Hard_disk",
            "e_Ambijentalno_osvetljenje",
            "e_Adaptivni_tempomat",
            "e_Digitalni_radio",
            "e_CD_changer",
            "e_Elektro_sklopivi_retrovizori",
            "e_Automatsko_zatamnjivanje_retrovizora",
            "e_MP3",
            "e_Navigacija",
            "e_Kuka_za_vuču",
            "e_360_kamera",
            "s_Senzor_mrtvog_ugla",
            "s_Asistencija_praćenja_trake",
            "e_Glasovne_komande",
            "e_Zaključavanje_diferencijala",
            "e_Vazdušno_vešanje",
            "e_Hands_free",
            "s_Mehanička_zaštita",
            "e_Adaptivna_svetla",
            "e_Panorama_krov",
            "e_Start_stop_sistem",
            "e_Brisači_prednjih_farova",
            "e_Elektro_zatvaranje_prtljažnika",
            "e_Asistencija_za_kretanje_na_uzbrdici",
            "e_USB",
            "s_Airbag_za_vozača",
            "s_Centralno_zaključavanje",
            "e_Bluetooth",
            "e_Električni_podizači",
            "e_DPF_filter",
            "s_OBD_zaštita",
            "s_Ulazak_bez_ključa",
            "e_Grejači_vetrobranskog_stakla",
            "e_Sportsko_vešanje",
            "e_Kamera",
            "e_Ručice_za_menjanje_brzina_na_volanu",
            "s_Blokada_motora",
            "e_Virtuelna_tabla",
            "e_AUX_konekcija",
            "s_Airbag_za_suvozača",
            "e_Senzori_za_kišu",
            "e_Ekran_na_dodir",
            "s_Kodiran_ključ",
            "e_Sedišta_podesiva_po_visini",
            "e_Putni_računar",
            "e_Servo_volan",
            "e_Senzori_za_svetla",
            "o_Kupljen_nov_u_Srbiji",
            "e_Podešavanje_volana_po_visini",
            "o_Garancija",
            "o_Prvi_vlasnik",
            "e_Grejači_retrovizora",
            "e_LED_zadnja_svetla",
            "e_Tempomat",
            "e_Parking_senzori",
            "e_LED_prednja_svetla",
            "e_Metalik_boja",
            "s_Alarm",
            "e_Branici_u_boji_auta",
            "e_Svetla_za_maglu",
            "e_Multifunkcionalni_volan",
            "e_Radio_CD",
            "e_Kožni_volan",
            "e_Naslon_za_ruku",
            "e_Tonirana_stakla",
            "e_Radio_ili_Kasetofon",
            "ai_sales_method",
            "e_Daljinsko_zaključavanje",
            "e_Elektro_podesiva_sedišta",
            "e_Utičnica_od_12V",
            "e_Grejanje_sedišta",
            "e_Električni_retrovizori",
            "e_Indikator_niskog_pritiska_u_gumama",
            "e_Paljenje_bez_ključa",
            "o_Servisna_knjiga",
            "s_Automatsko_kočenje",
            "e_Držači_za_čaše",
            "e_ISOFIX_sistem",
            "s_Child_lock",
            "s_ABS",
            "o_Rezervni_ključ",
            "e_Modovi_vožnje",
            "s_Bočni_airbag",
            "e_Krovni_nosač",
            "s_Vazdušni_jastuci_za_kolena",
            "e_Ostava_sa_hlađenjem",
            "ai_floating_flywheel",
            "e_Elektro_otvaranje_prtljažnika",
            "e_Dnevna_svetla",
            "ai_import_country",
            "e_Xenon_svetla",
            "e_Rezervni_točak",
            "gi_trade_in",
            "e_Aluminijumske_felne",
            "e_Multimedija",
            "ai_interior_color",
        ]
        features_info["features_to_delete"].extend(low_importance_features)

        return df, metadata

    @preprocess_init
    def clean(self, df: Dataset, metadata: Metadata) -> tuple[Dataset, Metadata]:
        if self.finalize_flag:
            df, metadata = MACleaner.ma_finalize(df=df, metadata=metadata)

        return df, metadata

    def start(self, df: Dataset, metadata: Metadata) -> tuple[Dataset, Metadata]:
        return self.clean(df, metadata)
