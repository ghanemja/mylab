"""Month names, weekday names and the AM/PM marker in the languages datetime understands.

Each entry has, in calendar order (weekdays from Sunday):
  months      - month names as used next to a day number (the format form; for Polish and Russian the
                genitive, as in "1 stycznia")
  months_sa   - month names on their own (the stand-alone form), when it differs
  months_abbr - abbreviated month names
  days        - weekday names
  days_abbr   - abbreviated weekday names
  ampm        - the markers for morning and afternoon
"""

from __future__ import annotations

_EN = {
    "months": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December"],
    "months_abbr": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "days": ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
    "days_abbr": ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
    "ampm": ["AM", "PM"],
}

LOCALES: dict[str, dict] = {
    "en": _EN,
    "fr": {
        "months": ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
                   "novembre", "décembre"],
        "months_abbr": ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.",
                        "déc."],
        "days": ["dimanche", "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi"],
        "days_abbr": ["dim.", "lun.", "mar.", "mer.", "jeu.", "ven.", "sam."],
        "ampm": ["AM", "PM"],
    },
    "de": {
        "months": ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober",
                   "November", "Dezember"],
        "months_abbr": ["Jan.", "Feb.", "März", "Apr.", "Mai", "Juni", "Juli", "Aug.", "Sept.", "Okt.", "Nov.",
                        "Dez."],
        "days": ["Sonntag", "Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag"],
        "days_abbr": ["So.", "Mo.", "Di.", "Mi.", "Do.", "Fr.", "Sa."],
        "ampm": ["AM", "PM"],
    },
    "es": {
        "months": ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
                   "noviembre", "diciembre"],
        "months_abbr": ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sept", "oct", "nov", "dic"],
        "days": ["domingo", "lunes", "martes", "miércoles", "jueves", "viernes", "sábado"],
        "days_abbr": ["dom", "lun", "mar", "mié", "jue", "vie", "sáb"],
        "ampm": ["a. m.", "p. m."],
    },
    "it": {
        "months": ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre",
                   "ottobre", "novembre", "dicembre"],
        "months_abbr": ["gen", "feb", "mar", "apr", "mag", "giu", "lug", "ago", "set", "ott", "nov", "dic"],
        "days": ["domenica", "lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato"],
        "days_abbr": ["dom", "lun", "mar", "mer", "gio", "ven", "sab"],
        "ampm": ["AM", "PM"],
    },
    "pt": {
        "months": ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro",
                   "outubro", "novembro", "dezembro"],
        "months_abbr": ["jan.", "fev.", "mar.", "abr.", "mai.", "jun.", "jul.", "ago.", "set.", "out.", "nov.",
                        "dez."],
        "days": ["domingo", "segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado"],
        "days_abbr": ["dom.", "seg.", "ter.", "qua.", "qui.", "sex.", "sáb."],
        "ampm": ["AM", "PM"],
    },
    "nl": {
        "months": ["januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september",
                   "oktober", "november", "december"],
        "months_abbr": ["jan", "feb", "mrt", "apr", "mei", "jun", "jul", "aug", "sep", "okt", "nov", "dec"],
        "days": ["zondag", "maandag", "dinsdag", "woensdag", "donderdag", "vrijdag", "zaterdag"],
        "days_abbr": ["zo", "ma", "di", "wo", "do", "vr", "za"],
        "ampm": ["a.m.", "p.m."],
    },
    "sv": {
        "months": ["januari", "februari", "mars", "april", "maj", "juni", "juli", "augusti", "september", "oktober",
                   "november", "december"],
        "months_abbr": ["jan.", "feb.", "mars", "apr.", "maj", "juni", "juli", "aug.", "sep.", "okt.", "nov.",
                        "dec."],
        "days": ["söndag", "måndag", "tisdag", "onsdag", "torsdag", "fredag", "lördag"],
        "days_abbr": ["sön", "mån", "tis", "ons", "tors", "fre", "lör"],
        "ampm": ["fm", "em"],
    },
    "pl": {
        "months": ["stycznia", "lutego", "marca", "kwietnia", "maja", "czerwca", "lipca", "sierpnia", "września",
                   "października", "listopada", "grudnia"],
        "months_sa": ["styczeń", "luty", "marzec", "kwiecień", "maj", "czerwiec", "lipiec", "sierpień", "wrzesień",
                      "październik", "listopad", "grudzień"],
        "months_abbr": ["sty", "lut", "mar", "kwi", "maj", "cze", "lip", "sie", "wrz", "paź", "lis", "gru"],
        "days": ["niedziela", "poniedziałek", "wtorek", "środa", "czwartek", "piątek", "sobota"],
        "days_abbr": ["niedz.", "pon.", "wt.", "śr.", "czw.", "pt.", "sob."],
        "ampm": ["AM", "PM"],
    },
    "ru": {
        "months": ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября",
                   "ноября", "декабря"],
        "months_sa": ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь",
                      "ноябрь", "декабрь"],
        "months_abbr": ["янв.", "февр.", "мар.", "апр.", "мая", "июн.", "июл.", "авг.", "сент.", "окт.", "нояб.",
                        "дек."],
        "days": ["воскресенье", "понедельник", "вторник", "среда", "четверг", "пятница", "суббота"],
        "days_abbr": ["вс", "пн", "вт", "ср", "чт", "пт", "сб"],
        "ampm": ["AM", "PM"],
    },
    "ja": {
        "months": [f"{k}月" for k in range(1, 13)],
        "months_abbr": [f"{k}月" for k in range(1, 13)],
        "days": ["日曜日", "月曜日", "火曜日", "水曜日", "木曜日", "金曜日", "土曜日"],
        "days_abbr": ["日", "月", "火", "水", "木", "金", "土"],
        "ampm": ["午前", "午後"],
    },
    "zh": {
        "months": ["一月", "二月", "三月", "四月", "五月", "六月", "七月", "八月", "九月", "十月", "十一月", "十二月"],
        "months_abbr": [f"{k}月" for k in range(1, 13)],
        "days": ["星期日", "星期一", "星期二", "星期三", "星期四", "星期五", "星期六"],
        "days_abbr": ["周日", "周一", "周二", "周三", "周四", "周五", "周六"],
        "ampm": ["上午", "下午"],
    },
}

DEFAULT_LOCALE = "en_US"  # used when no Locale is given
