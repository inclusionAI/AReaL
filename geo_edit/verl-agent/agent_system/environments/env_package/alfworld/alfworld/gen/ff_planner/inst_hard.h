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
 * File: inst_hard.h
 * Description: headers for multiplying hard operators.
 *
 *
 * Author: Joerg Hoffmann 2000
 *
 *********************************************************************/ 








#ifndef _INST_HARD_H
#define _INST_HARD_H



void build_hard_action_templates( void );



void cleanup_hard_domain( void );
Bool var_used_in_literals( int code_var, Literal *ef );
Bool var_used_in_numeric_effects( int code_var, NumericEffect *ef );
void decrement_inferior_vars_in_literals( int var, Literal *ef );
void decrement_inferior_vars_in_numeric_effects( int var, NumericEffect *ef );



void multiply_hard_op_parameters( void );
void create_hard_mixed_operators( Operator *o, int curr_var );
Effect *instantiate_Effect( Effect *e );
WffNode *instantiate_wff( WffNode *w );
void instantiate_exp( ExpNode **n );
Bool full_possibly_positive( Fact *f );
Bool full_possibly_negative( Fact *f );
int instantiated_fact_adress( Fact *f );



void multiply_hard_effect_parameters( void );
void create_hard_pseudo_effects( PseudoAction *a, Effect *e, int curr_var );
void make_instantiate_literals( PseudoActionEffect *e, Literal *ll );
void make_instantiate_numeric_effects( PseudoActionEffect *e, NumericEffect *ne );



#endif /* _INST_HARD_H */
