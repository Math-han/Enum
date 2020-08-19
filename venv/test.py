# -*- coding: utf-8 -*-
import pandas as pd
from enum_string import EnumString
from enum_mapper import EnumMapper

df_test = pd.DataFrame({'col1': ['1', '2', '3', '4', '5', 'o', '0'],
                        'col2': [0, 1, 2, 3, 4, 5, 6],
                        'col3': ['a', 'b', 'c', 'c', '6', '8', '8'],
                        'col4': [' 1', '2', ' 3', ' 4', ' 5', 'o', ' 0'],
                        'col5': [' .1', '2%', ' 3.33%', ' 4.%', ' .5%%  ', '00', ' 0'],
                        })

# Enum = EnumString(max_num_ratio=0.7)
# Enum.fit(df_test)
# df_test_enum = Enum.transform(df_test)
# print(df_test)
# print(df_test_enum)

Enum_2 = EnumMapper(maximum_enum_num=70)
Enum_2.fit(df_test)
df_test_enum_2 = Enum_2.transform(df_test)
print(df_test)
print(df_test_enum_2)

