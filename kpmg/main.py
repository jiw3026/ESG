import streamlit as st
from PIL import Image 



tab1, tab2, tab3 = st.tabs(['Environment' , 'Social','Governance'])

with tab1:
  st.write('기후 변화와 자원고갈의 위협이 커진 만큼, 투자자들은 지속 가능성 이슈를 투자 선택에 반영할 수 있습니다.')
    
with tab2:
  st.write('기업의 인권 보장과 데이터 보호,다양성의 고려,공급망 및 지역사회와의 협력관계 구축에 대한 정보를 알려드립니다.')

with tab3:
  st.write('환경과 사회 가치를 기업이 실현할 수 있도록 뒷받침하는 투명하고 신뢰도 높은 이사회 구성 및 감사위원회 구축에 대해 알려드립니다.')

  
st.title('TEAM OCTOPUS')

col1,col2 = st.columns([2,3])

with col1 :
    st.title('About ESG')
  
st.title('E')
st.write("""기후 변화와 자원고갈의 위협이 커진 만큼, 투자자들은 지속 가능성 이슈를 투자 선택에 반영할 수 있습니다.
""")

st.title('S')
st.write("""기업의 인권 보장과 데이터 보호,다양성의 고려,공급망 및 지역사회와의 협력관계 구축에 대한 정보를 알려드립니다.

""")
st.title('G')
st.write("""환경과 사회 가치를 기업이 실현할 수 있도록 뒷받침하는 투명하고 신뢰도 높은 이사회 구성과 감사위원회 구축 필요

""")



  


