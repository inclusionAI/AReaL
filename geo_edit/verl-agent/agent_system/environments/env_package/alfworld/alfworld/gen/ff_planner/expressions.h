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
 * File: expressions.h
 * Description: headers for handling numerical expressions
 *
 * Author: Joerg Hoffmann 2001
 *
 *********************************************************************/ 














#ifndef _EXPRESSIONS_H
#define _EXPRESSIONS_H




Bool number_comparison_holds( Comparator c, float l, float r );



Bool transform_to_LNF( void );
Bool is_linear_task( void );
Bool is_linear_expression( ExpNode *n );
void print_lnf_representation( void );



void normalize_expressions( void );
Bool translate_divisions( ExpNode **n );
void push_multiplications_down( ExpNode **n );
void put_comp_into_normalized_locals( Comparator comp,
				      ExpNode *lh,
				      ExpNode *rh );
void collect_normalized_locals( ExpNode *n, Bool positive );



void translate_subtractions( void );
Bool ex_fl_in_nF_of_pre_cond_eff_goal( int *fl );
void introduce_minus_fluent( int fl );
void replace_fl_in_nF_with_minus_fl( int fl );
void set_minus_fl_initial( int fl );
void introduce_minus_fl_effects( int fl );



void summarize_effects( void );
Bool same_condition( ActionEffect *e, ActionEffect *e_ );
Bool same_lnfs( LnfExpNode *l, LnfExpNode *r );
void merge_effects( ActionEffect *e, ActionEffect *e_ );
void merge_lnfs( LnfExpNode *l, LnfExpNode *r );



void encode_lfns_as_artificial_fluents( void );
Bool ex_non_minimal_lnf_in_pre_cond_goal_eff( void );
void introduce_artificial_fluent( void );
void replace_non_minimal_lnf_with_artificial_fl( void );
Bool is_artificial_fluent( LnfExpNode *n );



Bool setup_effect_costs( void );



void check_assigncycles( void );
Bool i_influences_j( int fi, int fj );
void determine_fl_relevance( void );
Bool i_inc_influences_j( int fi, int fj );



#endif /* _EXPRESSIONS_H */
