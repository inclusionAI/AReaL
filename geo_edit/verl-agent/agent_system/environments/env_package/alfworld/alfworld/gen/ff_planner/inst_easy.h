/*********************************************************************
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 * 
 *********************************************************************/






/*********************************************************************
 * File: inst_easy.h
 * Description: headers for multiplying easy operators.
 *
 *
 * Author: Joerg Hoffmann 2000
 *
 *********************************************************************/ 








#ifndef _INST_EASY_H
#define _INST_EASY_H



void build_easy_action_templates( void );



void cleanup_easy_domain( void );
Bool identical_fact( Fact *f1, Fact *f2 );
void handle_empty_easy_parameters( void );



void encode_easy_unaries_as_types( void );
int create_intersected_type( TypeArray T, int num_T );
int find_intersected_type( TypeArray T, int num_T );



void multiply_easy_effect_parameters( void );
void unify_easy_inertia_conditions( int curr_inertia );
void multiply_easy_non_constrained_effect_parameters( int curr_parameter );



void multiply_easy_op_parameters( void );
void unify_easy_inertia_preconds( int curr_inertia );
void multiply_easy_non_constrained_op_parameters( int curr_parameter );



#endif /* _INST_EASY_H */
