import React from 'react';
import Navbar from 'react-bootstrap/Navbar';

/**
 * Simple title bar placed at the top of the app.
 */
const NavBar = () => (
    <Navbar style={{ backgroundColor: "#173D4E", height: "15vh" }} expand="lg">
        <Navbar.Brand data-testid="navbar" style={{ fontSize: "4vh" }} color="white" href="#home" className="ml-5 text-white">Tackling Crohn's disease using Deep Learning</Navbar.Brand>
    </Navbar>
);

export default NavBar;