import streamlit as st
import pandas as pd
from model import open_data, preprocess_data, get_importances, predict_on_input
from PIL import Image


def preload_content():
    """ preload content used in web app """

    _, _, _, _, scaler = preprocess_data(open_data('data/clients.csv'))

    plane = Image.open('data/plane.jpg')
    age = Image.open('data/Age.png')
    cat = Image.open('data/Categorical.png')
    dist = Image.open('data/Distance.png')
    delays = Image.open('data/Delays.png')
    scores = Image.open('data/Scores.png')

    return scaler, plane, age, cat, dist, delays, scores


def highlight_weighs(s):
    """ generate colors to highlight weights """

    return ['background-color: #E6F6E4']*len(s) if s['Вес'] > 0 else ['background-color: #F6EBE4']*len(s)


def pack_input(sex, age, loyalty, distance, p_class, travel_type, dep_delay, arr_delay, wifi, fun,
               time_conv, onboard_service, booking, leg, gate_loc, baggage, food, checkin,
               online_boarding, inflight_service, seat, cleanliness):
    """ translate input values to pass to model """

    rule = {'Женский': 1,
            'Мужской': 0,
            'Личная': 0,
            'По работе': 1,
            'Лояльный': 1,
            'Нелояльный': 0}

    data = {'Gender': rule[sex],
            'Age': age,
            'Flight Distance': distance,
            'Departure Delay in Minutes': dep_delay,
            'Arrival Delay in Minutes': arr_delay,
            'Inflight wifi service': wifi,
            'Departure/Arrival time convenient': time_conv,
            'Ease of Online booking': booking,
            'Gate location': gate_loc,
            'Food and drink': food,
            'Online boarding': online_boarding,
            'Seat comfort': seat,
            'Inflight entertainment': fun,
            'On-board service': onboard_service,
            'Leg room service': leg,
            'Baggage handling': baggage,
            'Checkin service': checkin,
            'Inflight service': inflight_service,
            'Cleanliness': cleanliness,
            'Loyalty': rule[loyalty],
            'Business_travel': rule[travel_type],
            'Eco': 1 if p_class == 'Эко' else 0,
            'Eco Plus': 1 if p_class == 'Эко плюс' else 0}

    return pd.DataFrame(data, index=[0])


def render_page(scaler, plane, age, cat, dist, delays, scores):
    """ creates app page with tabs """

    st.title('Удовлетворённость полётом')
    st.subheader('Исследуем оценки, предсказываем удовлетворённость, оцениваем важность факторов')
    st.write('Материал - данные пассажиров и постполётные опросы')
    st.image(plane)

    tab1, tab2, tab3 = st.tabs([':mag: Исследовать', ':mage: Предсказать', ':vertical_traffic_light: Оценить'])

    with tab1:
        st.write('Exploratory data analysis: исследуем наши данные, предварительно очищенные и обработанные :sparkles:')

        st.write('**Возраст пассажиров**')
        st.image(age)
        st.write('Самый распространённый возраст - от 20 до 60 лет')
        st.divider()

        st.write('**Категориальные признаки:** пол, лояльность, тип поездки и удовлетворённость')
        st.image(cat)
        st.write('Пассажиры примерно равно распределены по полу')
        st.write('При этом лояльных и путешествующих по работе больше, чем нелояльных и путешествующих по личным нуждам')
        st.write('Довольных меньше, чем нейтральных и недовольных')
        st.divider()

        st.write('**Расстояние полёта**')
        st.image(dist)
        st.write('Самые частые полёты расстоянием до 1000 миль')
        st.divider()

        st.write('**Задержка вылета и прилёта**')
        st.image(delays)
        st.write('Задержки вылета и прилёта крайне редки и не превышают трёх часов')
        st.divider()

        st.write('**Постполётные опросы** (14 вопросов по шкале от 0 до 5)')
        st.image(scores)
        st.write('Оценки по опросам сдвинуты влево, самая популярная оценка - 4')
        st.write('Минимальных оценок достаточно мало')
        st.divider()

    with tab2:
        st.write('Введите данные вашего пассажира:')

        col1, col2, col3 = st.columns(3)
        with col1:
            sex = st.selectbox('Пол', ['Женский', 'Мужской'])
            travel_type = st.selectbox('Тип поездки', ['Личная', 'По работе'])
        with col2:
            age = st.slider('Возраст', min_value=0, max_value=100)
            p_class = st.selectbox('Класс билета', ['Бизнес', 'Эко', 'Эко плюс'])
        with col3:
            loyalty = st.selectbox('Лояльность', ['Лояльный', 'Нелояльный'])
            distance = st.slider('Расстояние полёта', min_value=100, max_value=3500)
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            dep_delay = st.slider('Задержка вылета, мин', min_value=0, max_value=500)
        with col2:
            arr_delay = st.slider('Задержка прилёта, мин', min_value=0, max_value=500)
        st.divider()

        st.write('Заполните опрос:')
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        with col1:
            wifi = st.radio('Интернет на борту', [0, 1, 2, 3, 4, 5])
            fun = st.radio('Развлечения на борту', [0, 1, 2, 3, 4, 5])
        with col2:
            time_conv = st.radio('Время вылета и прилёта', [0, 1, 2, 3, 4, 5])
            onboard_service = st.radio('Обслуживание на посадке', [0, 1, 2, 3, 4, 5])
        with col3:
            booking = st.radio('Онлайн-бронирование', [0, 1, 2, 3, 4, 5])
            leg = st.radio('Место в ногах', [0, 1, 2, 3, 4, 5])
        with col4:
            gate_loc = st.radio('Расположение выхода на посадку', [0, 1, 2, 3, 4, 5])
            baggage = st.radio('Обращение с багажом', [0, 1, 2, 3, 4, 5])
        with col5:
            food = st.radio('Еда и напитки на борту', [0, 1, 2, 3, 4, 5])
            checkin = st.radio('Регистрация на рейс', [0, 1, 2, 3, 4, 5])
        with col6:
            online_boarding = st.radio('Онлайн выбор места', [0, 1, 2, 3, 4, 5])
            inflight_service = st.radio('Обслуживание на борту', [0, 1, 2, 3, 4, 5])
        with col7:
            seat = st.radio('Удобство сиденья', [0, 1, 2, 3, 4, 5])
            cleanliness = st.radio('Чистота на борту', [0, 1, 2, 3, 4, 5])

        col1, col2, col3 = st.columns(3)
        if col2.button('Полетели!'):
            with st.spinner('Считаем!'):
                inputs = pack_input(sex, age, loyalty, distance, p_class, travel_type, dep_delay, arr_delay, wifi, fun,
                                    time_conv, onboard_service, booking, leg, gate_loc, baggage, food, checkin,
                                    online_boarding, inflight_service, seat, cleanliness)
                scaled = pd.DataFrame(scaler.transform(inputs), columns=inputs.columns)

                pred, proba = predict_on_input(scaled)
                if pred == 1:
                    st.success('Пассажир доволен! :thumbsup: :thumbsdown:')
                    with st.expander('Подробнее'):
                        st.write(f'Вероятность этого: **`{round(max(proba[0]), 3)}`**')
                elif pred == 0:
                    st.error('Пассажир не доволен :thumbsdown: :thumbsdown:')
                    with st.expander('Подробнее'):
                        st.write(f'Вероятность этого: **`{round(max(proba[0]), 3)}`**')
                else:
                    st.error('Что-то пошло не так...')

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.write('**Что важно для хорошего впечатления пассажира?**')
            st.dataframe(get_importances(5, 'most').style.apply(highlight_weighs, axis=1))
        with col2:
            st.write('**А что практически не важно?**')
            st.dataframe(get_importances(5, 'least').style.apply(highlight_weighs, axis=1))


def load_page():
    """ loads main page """

    scaler, plane, age, cat, dist, delays, scores = preload_content()

    st.set_page_config(layout="wide",
                       page_title="Полёты и опросы",
                       page_icon=':airplane:')

    render_page(scaler, plane, age, cat, dist, delays, scores)


if __name__ == "__main__":
    load_page()
